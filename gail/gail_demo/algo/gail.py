import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

from .ppo import PPO
from ..buffer import RolloutBuffer
from gail_demo.network import GAILDiscrim, VDB, LSTMCoder

import wandb
import pdb
import random

from transforms3d.euler import euler2mat, euler2quat
from transforms3d.quaternions import mat2quat

def deg2quat(deg_states):
    state = np.array([])
    states = []
    if len(deg_states.shape) == 1:
        for i in range(deg_states.shape[0]):
            hip_joint = [0, 0, deg_states[i]]
            quat = euler2quat(*hip_joint)
            state = np.concatenate((state, quat))
        states.append(state.tolist())
    
    else:
        for i in range(deg_states.shape[0]):
            state = np.array([])
            for j in range(deg_states.shape[1]):
                hip_joint = [0, 0, deg_states[i][j]]
                quat = euler2quat(*hip_joint)
                state = np.concatenate((state, quat))
            # pdb.set_trace()
            states.append(state.tolist())
    
    states = torch.Tensor(states)
    return states
    
def deg2sixdrr(deg_states):
    state = np.array([])
    states = []
    if len(deg_states.shape) == 1:
        for i in range(deg_states.shape[0]):
            hip_joint = [0, 0, deg_states[i]]
            mat = euler2mat(*hip_joint)[:,:2].flatten()
            state = np.concatenate((state, mat))
        states.append(state.tolist())
    
    else:
        for i in range(deg_states.shape[0]):
            state = np.array([])
            for j in range(deg_states.shape[1]):
                hip_joint = [0, 0, deg_states[i][j]]
                mat = euler2mat(*hip_joint)[:,:2].flatten()
                state = np.concatenate((state, mat))
            # pdb.set_trace()
            states.append(state.tolist())
    
    states = torch.Tensor(states)
    return states

def calculate_adv(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    return deltas + values, (deltas - deltas.mean()) / (deltas.std() + 1e-8)

def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class Chunking(): 
    def __init__(self, state_shape=18, action_shape=4, obs_history=4, act_chunked=4, device=None):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.obs_history = obs_history
        self.act_chunked = act_chunked
        self.obs_chunker = torch.zeros((obs_history, state_shape), device=device)
        self.act_chunker = np.zeros((act_chunked, 2 * act_chunked - 1, action_shape))
        self.device = device

        self.k = 0.01
        self.exp_weights = np.exp(-self.k * np.arange(self.act_chunked))
        self.exp_weights = self.exp_weights / self.exp_weights.sum()
        self.exp_weights = self.exp_weights[:, np.newaxis]
    def obs_trunker_init(self, state:torch.tensor): 
        self.obs_chunker[:] = state
    def obs_trunker_update(self, state:torch.tensor): 
        for i in range(self.obs_history - 1):
            self.obs_chunker[i] = self.obs_chunker[i + 1]
        self.obs_chunker[-1] = state
    def get_obs(self): 
        return self.obs_chunker.clone().flatten()
    

    def act_trunker_init(self, k_actions:np.ndarray):
        k_actions_reshaped = k_actions.reshape(-1, self.action_shape)
        for i in range(self.act_chunked): 
            self.act_chunker[i, i:i + self.act_chunked] = k_actions_reshaped

        # print(self.act_chunker)
        # pdb.set_trace()

    def act_trunker_update(self, k_actions:np.ndarray):
        k_actions_reshaped = k_actions.reshape(-1, self.action_shape)
        for i in range(self.act_chunked - 1): 
            self.act_chunker[i, i:i + self.act_chunked] = self.act_chunker[i+1, i+1:i+self.act_chunked+1]
        self.act_chunker[-1, self.act_chunked-1:] = k_actions_reshaped

        # print(self.act_chunker)
        # pdb.set_trace()
    def get_act(self): 
        actions_history = self.act_chunker[:, self.act_chunked-1]
        weighted_actions_history = self.exp_weights * actions_history
        return weighted_actions_history.sum(axis=0)


def obs_chunk(observations:torch.Tensor, N=4): 
    '''
    Originally observations are encoded with onehot vector at each time step t.
    Now transform the batched observations to chunked observation with history O_{t} to O_{t-N+1:t}
    '''
    assert N > 0
    if N == 1:
        return observations
    else:
        chunked_obs = torch.zeros((observations.shape[0], observations.shape[1] * N), dtype=torch.float32, device=observations.device)
        chunked_obs[:N, :] = observations[0].repeat(N, N)
        chunked_obs[:, observations.shape[1] * (N - 1):] = observations
        for i in reversed(range(N - 1)):
            chunked_obs[N - i - 1:, i * observations.shape[1]: (i+1) * observations.shape[1]] = observations[:i + 1 - N]
        return chunked_obs

def token_duplicated(token: torch.Tensor, N=1):
    '''
    This is for duplicating an env_id so as to match the states before concatenating
    Note that here the size should be in 3D
    '''
    assert len(token.shape) == 3
    if N > 1:
        tslice = token[:, -1:, :]
        for i in range(N - 1):
            token = torch.cat((token, tslice), dim=1)
    return token



class GAIL(PPO):

    def __init__(self, buffer_exp, state_shape, action_shape, muscle_shape, device, seed,
                 gamma=0.995, rollout_length=50000, mix_buffer=1,
                 batch_size_disc=64, batch_size_ppo=10000, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-4,
                 units_actor=(64, 64), units_critic=(64, 64),
                 units_disc=(100, 100), epoch_ppo=50, epoch_disc=10,
                 clip_eps=0.2, lambd=0.97, coef_ent=0.0, max_grad_norm=10.0, header_num=0, obs_history=1, act_chunked=1, num_steps=10**5, 
                 units_vdb=(64, 64), z_dim_vdb=[8], beta=0.1, dual_stepsize=1e-5, mutual_info_constr=0.5, lstm_dim=16, lstm_layers=2, lstm_outputdim=[8]):
        super().__init__(
            state_shape, action_shape, muscle_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm, header_num, obs_history, act_chunked
        )



        # Expert's buffer.
        self.buffer_exp = buffer_exp
        self.seed = seed

        # Agent buffer
        self.buffer_hamwl = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape[0],
            action_shape=action_shape[0]*act_chunked,
            muscle_shape=muscle_shape, 
            device=device,
            mix=mix_buffer
        )
        self.buffer_iliowl = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape[0],
            action_shape=action_shape[0]*act_chunked,
            muscle_shape=muscle_shape, 
            device=device,
            mix=mix_buffer
        )
        self.buffer_shortham = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape[0],
            action_shape=action_shape[0]*act_chunked,
            muscle_shape=muscle_shape, 
            device=device,
            mix=mix_buffer
        )
        self.buffer_all = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape[0],
            action_shape=action_shape[0]*act_chunked,
            muscle_shape=muscle_shape, 
            device=device,
            mix=mix_buffer * 3
        )
        self.buffer_sum = {'sconewalk_origin_motored_hamstringweakness_new_h0914-v3': self.buffer_hamwl, 
                           'sconewalk_origin_motored_iliopsoasweakness_h0914-v3': self.buffer_iliowl, 
                           'sconewalk_origin_motored_shorthamstring_h0914-v3': self.buffer_shortham, }
        
        self.env_id_dict = {'sconewalk_origin_motored_hamstringweakness_new_h0914-v3': torch.from_numpy(np.array([0])), 
                           'sconewalk_origin_motored_iliopsoasweakness_h0914-v3': torch.from_numpy(np.array([1])), 
                           'sconewalk_origin_motored_shorthamstring_h0914-v3': torch.from_numpy(np.array([2])), }
        
        

        # Discriminator.
        self.disc = GAILDiscrim(
            state_shape=state_shape,
            #action_shape=action_shape,
            hidden_units=units_disc,
            hidden_activation=nn.Tanh(), 
            header=header_num, 
            obs_history=obs_history
        ).to(device)

        # self.vdb = VDB(state_shape=state_shape, 
        #                hidden_units=units_vdb, 
        #                z_dim=z_dim_vdb, 
        #                hidden_activation=nn.Tanh(), 
        #                header=header_num, 
        #                obs_history=obs_history).to(device)
        
        self._beta = beta
        self._lr_beta = dual_stepsize
        self.mutual_info_constr = mutual_info_constr
        self.z_dim_vdb = z_dim_vdb

        # self.lstm = LSTMCoder(state_shape=state_shape, 
        #                     hidden_size=lstm_dim, 
        #                     output_size=lstm_outputdim, 
        #                     num_layers=lstm_layers, 
        #                     header=header_num, 
        #                     ).to(device)
        self.lstm_dim = lstm_dim
        self.lstm_layers = lstm_layers
        self.lstm_outputdim = lstm_outputdim

        self.learning_steps_disc = 0
        self.optim_disc = Adam([{'params': self.disc.parameters()}], lr=lr_disc) #, {'params': self.embed.parameters()}, {'params': self.vdb.parameters()}

        if False:
            self.embed = torch.nn.Embedding(3, 9)
            self.optim_disc = Adam([{'params': self.disc.parameters()}, {'params': self.embed.parameters()}], lr=lr_disc) #, {'params': self.embed.parameters()}
            self.optim_actor = Adam([{'params': self.actor.parameters()}, {'params': self.embed.parameters()}], lr=lr_actor)
            self.optim_critic = Adam([{'params': self.critic.parameters()}, {'params': self.embed.parameters()}], lr=lr_critic)
        
        self.batch_size_disc = batch_size_disc
        self.batch_size_ppo = batch_size_ppo
        self.epoch_disc = epoch_disc
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.obs_history = obs_history
        self.act_chunked = act_chunked
        self.chunker = Chunking(header_num + state_shape[0], action_shape[0], obs_history, act_chunked, device)
        # pdb.set_trace()

    def gradient_norm(self, network): 
        param_norm = 0
        total_norm = 0
        for p in network.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5



    def get_latent_kl_div(self, states):
        mu, log_sigma = self.vdb.get_z(states)
        # return torch.mean(-sigma+(torch.square(mu)+torch.square(torch.exp(sigma))-1.)/2.)
        kl_div = 0.5 * torch.sum(torch.pow(mu, 2) + torch.exp(log_sigma) - log_sigma - 1, dim=1)
        return kl_div.mean()


    def embed (self, i): 
        self.embed1 = torch.tensor([[0, 0, 1], 
                                   [0, 1, 0], 
                                   [1, 0, 0]])
        # pdb.set_trace()
        return self.embed1[i]
    

    def is_update(self, name):
        if self.buffer_sum[name]._p % self.rollout_length == 0 and not self.buffer_sum[name].clean:
            return name, True
        return None, False
        # return step % self.rollout_length == 0

    def step(self, name, env, state, t, step, init_state, norm=True):
        state_w_envid = torch.concat((self.embed(self.env_id_dict[name]).squeeze().detach(), torch.from_numpy(state)))
        # state_w_envid = torch.from_numpy(state)

        if t == 0: 
            self.chunker.obs_trunker_init(state_w_envid)
        else:
            self.chunker.obs_trunker_update(state_w_envid)

        state_for_policy = self.chunker.get_obs()

        k_actions, log_pi = self.explore(state_for_policy)
        # action = np.tanh(action)
        if t == 0: 
            self.chunker.act_trunker_init(k_actions)
        else:
            self.chunker.act_trunker_update(k_actions)
        action = self.chunker.get_act()
        # pdb.set_trace()
        next_state, next_state_ori, reward, done, info = env.step(action, norm=norm)
        t += 1
        '''
        if t % 100 == 0:
            for key in info['rew_dict'].keys():
                wandb.log({key: info['rew_dict'][key]}) # similar to this
        '''
        mask = False if t == env._max_episode_steps else done
        
        # next_state_for_store = torch.concat((self.embed(self.env_id_dict[name]).squeeze(), torch.from_numpy(next_state)))

        self.buffer_sum[name].append(state, k_actions, reward, mask, log_pi, next_state, info['rew_dict']['gaussian_vel'], info['rew_dict']['grf'], info['rew_dict']['constr'])
        

        if done:
            t = 0
            next_state, next_state_ori = env.reset(norm=norm, init_values=init_state, random_switch=True)

        return next_state, t, info, done

    def update(self, writer, env, name, config=None):
        self.learning_steps += 1
        mean = torch.from_numpy(env.obs_mean).to(self.device)
        var = torch.from_numpy(env.obs_var).to(self.device)


        for epoch in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories.
            # states, trunk_vels, actions = self.buffer_sum[name].sample(self.batch_size_disc)[:3]
            states, trunk_vels, actions = self.buffer_sum[name].sample_w_history(self.batch_size_disc, self.obs_history)[:3]
            # Samples from expert's demonstrations.
            # states_exp, trunk_vels_exp, actions_exp = self.buffer_exp[name].sample(self.batch_size_disc)[:3]
            states_exp, trunk_vels_exp, actions_exp = self.buffer_exp[name].sample_w_history(self.batch_size_disc, self.obs_history)[:3]

            # Embedded env_id for agent
            # token = self.embed(torch.full((self.batch_size_disc, 1), self.env_id_dict[name].item())).to(device=self.device).squeeze(1)
            token = self.embed(torch.full((self.batch_size_disc, 1), self.env_id_dict[name].item())).to(device=self.device)
            token = token_duplicated(token, self.obs_history)
            states = torch.cat((token, states), dim=-1)
            states_history = states.view(self.batch_size_disc, -1)

            # Normalize expert data
            states_exp = ((states_exp - mean) / torch.sqrt(var + 1e-8)).to(torch.float32)
            # Add embedding
            # token_exp = self.embed(torch.full((self.batch_size_disc, 1), self.env_id_dict[name].item())).to(device=self.device).squeeze(1).detach()
            token_exp = self.embed(torch.full((self.batch_size_disc, 1), self.env_id_dict[name].item())).to(device=self.device).detach()
            token_exp = token_duplicated(token_exp, self.obs_history)
            states_exp = torch.cat((token_exp, states_exp), dim=-1)
            states_exp_history = states_exp.view(self.batch_size_disc, -1)

            # Update discriminator.
            # pdb.set_trace()
            # self.update_disc(states, states_exp, epoch, writer)
            self.update_disc(states_history, states_exp_history, epoch, writer)
            wandb.log({"disc_gradient_norm": self.gradient_norm(self.disc)})
            # self.update_disc(quer_states, actions, quer_states_exp, actions_exp, writer)
            #self.update_disc(states, actions, states_exp, actions_exp, writer, next_states, next_states_exp) # For states + next_states

        # We don't use reward signals here. Depending on the situation, data could be collected from one buffer or sampled from all three
        # states, quer_states, actions, rewards_env, dones, log_pis, next_states, quer_next_states, trunk_vels, rew_grfs, rew_constrs = self.buffer_sum[name].get()
        states_full, _, actions_full, _, dones_full, log_pis_full, next_states_full, _, trunk_vels, rew_grfs, rew_constrs = self.buffer_sum[name].get_w_history(self.obs_history)
        states, actions, dones, log_pis, next_states = states_full, actions_full, dones_full, log_pis_full, next_states_full
        # for i in range(int(self.rollout_length / self.batch_size_ppo)): 
        #     states = states_full[i*self.batch_size_ppo:(i+1)*self.batch_size_ppo]
        #     actions = actions_full[i*self.batch_size_ppo:(i+1)*self.batch_size_ppo]
        #     dones = dones_full[i*self.batch_size_ppo:(i+1)*self.batch_size_ppo]
        #     log_pis = log_pis_full[i*self.batch_size_ppo:(i+1)*self.batch_size_ppo]
        #     next_states = next_states_full[i*self.batch_size_ppo:(i+1)*self.batch_size_ppo]

        # Calculate rewards.
        # token = self.embed(torch.full((states.shape[0], 1), self.env_id_dict[name].item())).to(device=self.device).squeeze(1).detach()
        token = self.embed(torch.full((states.shape[0], 1), self.env_id_dict[name].item())).to(device=self.device).detach()
        token = token_duplicated(token, self.obs_history)
        states_for_disc = torch.cat((token, states), dim=-1)
        # states_for_disc = states
        states_for_disc_history = states_for_disc.view(states_for_disc.shape[0], -1)
        # disc_rewards = self.disc.calculate_reward(states_for_disc)
        # disc_rewards = self.disc.calculate_reward(states_for_disc_chunked)
        disc_rewards = self.disc.calculate_reward(states_for_disc_history)
        rewards = disc_rewards 
        
        # token = self.embed(torch.full((self.batch_size_disc, 1), self.env_id_dict[name].item())).to(device=self.device).squeeze(1)
        with torch.no_grad():
            states_for_critics = torch.cat((token, states), dim=-1)
            # states_for_critics = states
            states_for_critics_history = states_for_critics.view(states_for_critics.shape[0], -1)
            next_states_for_critics = torch.cat((token, next_states), dim=-1)
            # next_states_for_critics = next_states
            next_states_for_critics_history = next_states_for_critics.view(next_states_for_critics.shape[0], -1)
            # values = self.critic(states_for_critics)
            # next_values = self.critic(next_states_for_critics)
            values = self.critic(states_for_critics_history)
            next_values = self.critic(next_states_for_critics_history)

        targets, gaes = calculate_gae(values, rewards, dones, next_values, self.gamma, self.lambd)
        # targets, gaes = calculate_adv(values, rewards, dones, next_values, self.gamma, self.lambd)

        for epoch in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            # token = self.embed(torch.full((states.shape[0], 1), self.env_id_dict[name].item())).to(device=self.device).squeeze(1)
            token = self.embed(torch.full((states.shape[0], 1), self.env_id_dict[name].item())).to(device=self.device)
            token = token_duplicated(token, self.obs_history)
            states_for_updates = torch.cat((token, states), dim=-1)
            # states_for_updates = states
            states_for_updates_history = states_for_updates.view(states_for_updates.shape[0], -1)
            # pdb.set_trace()
            # self.update_critic(states_for_updates, targets, epoch, writer)
            # self.update_actor(states_for_updates, actions, log_pis, gaes, epoch, writer)
            self.update_critic(states_for_updates_history, targets, epoch, writer)
            self.update_actor(states_for_updates_history, actions, log_pis, gaes, epoch, writer)
            wandb.log({"actor_gradient_norm": self.gradient_norm(self.actor)})
            wandb.log({"critic_gradient_norm": self.gradient_norm(self.critic)})
    

    def update_disc(self, states, states_exp, epoch, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        # TODO: With Sigmoid 2. MSE bet
        logits_pi = self.disc(states)
        logits_exp = self.disc(states_exp)

        # For VAIL
        # logits_pi = self.vdb(states)
        # logits_exp = self.vdb(states_exp)
        # logits_pi = self.disc(logits_pi)
        # logits_exp = self.disc(logits_exp)

        # For LSTM
        # pdb.set_trace()
        # logits_pi = self.lstm(states)
        # logits_exp = self.lstm(states_exp)
        # logits_pi = self.disc(logits_pi)
        # logits_exp = self.disc(logits_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()


        # VAIL's bottleneck
        # bottleneck_loss = 0.5 * (self.get_latent_kl_div(states) + self.get_latent_kl_div(states_exp)) - self.mutual_info_constr
        # gradient_penalty_vail = self._gradient_penalty_vail(states, states_exp, lamda=0.5, k=0)
        # self._beta = max(0, self._beta + self._lr_beta * bottleneck_loss.item())
        # loss_disc = loss_pi + loss_exp + self._beta * bottleneck_loss + gradient_penalty_vail


        gradient_penalty = self._gradient_penalty(states, states_exp, lamda=1.0, k=0)
        loss_disc = loss_pi + loss_exp + gradient_penalty
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

        # if self.learning_steps_disc % self.epoch_disc == 0:
        #     writer.add_scalar(
        #         'loss/disc', loss_disc.item(), self.learning_steps)

        #     # Discriminator's accuracies.
        #     with torch.no_grad():
        #         acc_pi = (logits_pi < 0).float().mean().item()
        #         acc_exp = (logits_exp > 0).float().mean().item()
        #     writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
        #     writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)
    

    def _gradient_penalty(self, states, states_exp, lamda, k): 
        #pdb.set_trace()
        epsilon = torch.rand(self.batch_size_disc, 1).to(self.device)
        states_hat = epsilon * states_exp + (1.0 - epsilon) * states 
        states_hat = Variable(states_hat, requires_grad=True)
        outputs_hat = self.disc(states_hat)
        # outputs_hat = torch.tanh(output_hat)
        gradients = torch_grad(outputs=outputs_hat, inputs=states_hat, 
                               grad_outputs=torch.ones(outputs_hat.size()).to(self.device), 
                               create_graph=True, retain_graph=True)[0]
        
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1))

        return lamda * ((gradients_norm - k) ** 2).mean()
    
    def _gradient_penalty_vail(self, states, states_exp, lamda=1.0, k=0): 
        # epsilon = torch.rand(self.batch_size_disc, 1).to(self.device)
        states_hat = states_exp
        states_hat = Variable(states_hat, requires_grad=True)
        mu, sigma = self.vdb.get_z(states_hat)
        outputs_hat = self.disc(mu + torch.randn_like(mu) * torch.exp(sigma))
        gradients = torch_grad(outputs=outputs_hat, inputs=states_hat, 
                               grad_outputs=torch.ones(outputs_hat.size()).to(self.device), 
                               create_graph=True, retain_graph=True)[0]
        
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1))

        return lamda * ((gradients_norm - k) ** 2).mean()
    
    def _gradient_penalty_lstm(self, states, states_exp, lamda, k): 
        #pdb.set_trace()
        epsilon = torch.rand(self.batch_size_disc, 1).to(self.device)
        states_hat = epsilon * states_exp + (1.0 - epsilon) * states 
        states_hat = Variable(states_hat, requires_grad=True)
        outputs_hat = self.disc(states_hat)
        # outputs_hat = torch.tanh(output_hat)
        gradients = torch_grad(outputs=outputs_hat, inputs=states_hat, 
                               grad_outputs=torch.ones(outputs_hat.size()).to(self.device), 
                               create_graph=True, retain_graph=True)[0]
        
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1))

        return lamda * ((gradients_norm - k) ** 2).mean()
    
    
    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states,
                   writer):
        
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)

        for epoch in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets, epoch, writer)
            self.update_actor(states, actions, log_pis, gaes, epoch, writer)

    def update_critic(self, states, targets, epoch, writer):
        #pdb.set_trace()
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()

        # if epoch == self.epoch_ppo - 1:
        #     loss_critic.backward()
        # else:
        #     loss_critic.backward(retain_graph=True)

        loss_critic.backward(retain_graph=True)

        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        # if self.learning_steps_ppo % self.epoch_ppo == 0:
        #     writer.add_scalar(
        #         'loss/critic', loss_critic.item(), self.learning_steps)

    def update_actor(self, states, actions, log_pis_old, gaes, epoch, writer):
        #pdb.set_trace()
        log_pis = self.actor.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        # pdb.set_trace()

        # if epoch == self.epoch_ppo - 1:
        #     (loss_actor - self.coef_ent * entropy).backward()
        # else:
        #     (loss_actor - self.coef_ent * entropy).backward(retain_graph=True)
        
        (loss_actor - self.coef_ent * entropy).backward()

        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        # if self.learning_steps_ppo % self.epoch_ppo == 0:
        #     writer.add_scalar(
        #         'loss/actor', loss_actor.item(), self.learning_steps)
        #     writer.add_scalar(
        #         'stats/entropy', entropy.item(), self.learning_steps)
        
