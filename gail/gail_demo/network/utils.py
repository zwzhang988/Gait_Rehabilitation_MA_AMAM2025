import math
import torch
from torch import nn
from torch.nn.utils import spectral_norm
import pdb
def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh(), output_activation=None, add_spectral=False):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layer = nn.Linear(units, next_units)
        if add_spectral:
            layer = spectral_norm(layer)
        layers.append(layer)
        layers.append(hidden_activation)
        units = next_units
    
    out_layer = nn.Linear(units, output_dim)
    if add_spectral:
        out_layer = spectral_norm(out_layer)
    layers.append(out_layer)
    
    if output_activation is not None:
        layers.append(output_activation)
    
    return nn.Sequential(*layers)


def calculate_log_pi(log_stds, noises, actions):
    #pdb.set_trace()
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(
        dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)
    
    return gaussian_log_probs

    # return gaussian_log_probs - torch.log(
    #     1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)


def reparameterize(means, log_stds):
    #pdb.set_trace()
    noises = torch.randn_like(means)
    us = means + noises * log_stds.exp()
    #actions = torch.tanh(us)
    actions = us
    return actions, calculate_log_pi(log_stds, noises, actions)


def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def evaluate_lop_pi(means, log_stds, actions):
    #pdb.set_trace()
    #noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    noises = (actions - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)
