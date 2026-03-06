import torch
from torch import nn
import torch.nn.functional as F

from .utils import build_mlp


class LSTMCoder(nn.Module):

    def __init__(self, state_shape, hidden_size, output_size, num_layers, header, ):
        super(LSTMCoder, self).__init__()

        self.lstm = nn.LSTM(state_shape[0] + header, hidden_size, num_layers)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2out = nn.Linear(hidden_size, output_size[0])

    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence.view(len(sentence), 1, -1))
        out_space = self.hidden2out(lstm_out.view(len(sentence), -1))
        out_scores = F.log_softmax(out_space, dim=1)
        return out_scores


class VDB(nn.Module):
    def __init__(self, state_shape, hidden_units=(100, 100), z_dim=[16], 
                 hidden_activation=nn.Tanh(), header=0, obs_history=1):
        super(VDB, self).__init__()
        # self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.mu = nn.Linear(hidden_dim, z_dim) 
        # self.sigma = nn.Linear(hidden_dim, z_dim) 

        self.net_mu = build_mlp(
            input_dim=(state_shape[0] + header) * obs_history,
            output_dim=z_dim[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
        )

        self.net_sigma = build_mlp(
            input_dim=(state_shape[0] + header) * obs_history,
            output_dim=z_dim[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
        )
    
    def forward(self, x):
        mu = self.net_mu(x)
        sigma = self.net_sigma(x)
        std = torch.exp(sigma/2)
        eps = torch.randn_like(std)
        return  mu + std * eps

    def get_z(self, x):
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        mu = self.net_mu(x)
        sigma = self.net_sigma(x)
        return mu, sigma
    
    def get_mean(self, x):
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        mu = self.net_mu(x)
        return mu



class GAILDiscrim(nn.Module):

    def __init__(self, state_shape, hidden_units=(100, 100),
                 hidden_activation=nn.Tanh(), header=0, obs_history=1):
        super().__init__()

        self.net = build_mlp(
            input_dim=(state_shape[0] + header) * obs_history,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
            add_spectral=True
        )

    def forward(self, states):
        return self.net(states)

    def calculate_reward(self, states):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        with torch.no_grad():
            return -F.logsigmoid(-self.forward(states))


class AIRLDiscrim(nn.Module):

    def __init__(self, state_shape, gamma,
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True)):
        super().__init__()

        self.g = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r
        )
        self.h = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v
        )

        self.gamma = gamma

    def f(self, states, dones, next_states):
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(self, states, dones, log_pis, next_states):
        # Discriminator's output is sigmoid(f - log_pi).
        return self.f(states, dones, next_states) - log_pis

    def calculate_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            return -F.logsigmoid(-logits)
