import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

from .ppo import PPO
from .gail import GAIL
from ..buffer import RolloutBuffer
from gail_demo.network import GAILDiscrim, VDB

import wandb
import pdb
import random

from transforms3d.euler import euler2mat, euler2quat
from transforms3d.quaternions import mat2quat





class VAIL(GAIL): 

    def __init__(self, buffer_exp, state_shape, action_shape, muscle_shape, device, seed,
                 gamma=0.995, rollout_length=50000, mix_buffer=1,
                 batch_size_disc=64, batch_size_ppo=10000, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-4,
                 units_actor=(64, 64), units_critic=(64, 64), units_disc=(100, 100), epoch_ppo=50, epoch_disc=10,
                 clip_eps=0.2, lambd=0.97, coef_ent=0.0, max_grad_norm=10.0, header_num=0, obs_chunked=1, act_chunked=1, num_steps=10**5, 
                 units_vdb=(64, 64), beta=0, dual_stepsize=3e-4, mutual_info_constr=0.1):
        
        super().__init__(buffer_exp, state_shape, action_shape, muscle_shape, device, seed, gamma, rollout_length, mix_buffer,
                 batch_size_disc, batch_size_ppo, lr_actor, lr_critic, lr_disc, units_actor, units_critic,
                 units_disc, epoch_ppo, epoch_disc, clip_eps, lambd, coef_ent, max_grad_norm, header_num, obs_chunked, act_chunked, num_steps)
        
        self.vdb = VDB(state_shape=state_shape, 
                       hidden_units=units_vdb, 
                       z_dim=16, 
                       hidden_activation=nn.Tanh()).to(device)
        
        self.beta = beta
        self.dual_stepsize = dual_stepsize
        self.mutual_info_constr = mutual_info_constr
    
    def get_latent_kl_div(self, states):
        _, mu, sigma = self.vdb.get_z(states)
        return torch.mean(-sigma+(torch.square(mu)+torch.square(torch.exp(sigma))-1.)/2.)

    def update_disc(self, states, states_exp, epoch, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        # TODO: With Sigmoid 2. MSE bet
        logits_pi = self.disc(states)
        logits_exp = self.disc(states_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        # gradient_penalty = self._gradient_penalty(states, states_exp, lamda=1.0, k=0)

        # VAIL's bottleneck
        bottleneck_loss = 0.5 * (self.get_latent_kl_div(states) + self.get_latent_kl_div(states_exp)) - self.mutual_info_constr
        self.beta = max(0, self.beta + self.dual_stepsize * bottleneck_loss.item())
        loss_disc = loss_pi + loss_exp + self.beta * bottleneck_loss
        # loss_disc = loss_pi + loss_exp + gradient_penalty
        # loss_disc = loss_pi + loss_exp
        wandb.log({"logits_pi": F.sigmoid(logits_pi).mean()})
        wandb.log({"logits_exp": F.sigmoid(logits_exp).mean()})
        wandb.log({"logits_diff": (F.sigmoid(logits_exp) - F.sigmoid(logits_pi)).mean()})

        # wandb.log({"gradient_penalty": gradient_penalty})

        # TODO: GRAD PENALTY GOES HERE
        # try target_grad = 0/1

        self.optim_disc.zero_grad()

        # if epoch == self.epoch_disc - 1:
        #     loss_disc.backward()
        # else:
        #     loss_disc.backward(retain_graph=True)
        loss_disc.backward()

        self.optim_disc.step()