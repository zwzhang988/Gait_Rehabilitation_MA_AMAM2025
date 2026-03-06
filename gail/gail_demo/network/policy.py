import torch
from torch import nn
import numpy as np
import pdb
from .utils import build_mlp, reparameterize, evaluate_lop_pi
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

class StateIndependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh(), header=0, obs_history=1, act_chunked=1):
        super().__init__()

        self.net = build_mlp(
            input_dim=(state_shape[0] + header) * obs_history,
            output_dim=action_shape[0] * act_chunked,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.fill_(0.01)

        self.net.apply(init_weights)

        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0] * act_chunked))

    def forward(self, states):
        # return torch.tanh(self.net(states))
        # states = deg2quat(states).to('cuda:0') # for quaternion
        # states = deg2sixdrr(states).to('cuda:0') # for rotation matrix
        return self.net(states)

    def sample(self, states):
        # states = deg2quat(states).to('cuda:0') # for quaternion
        # states = deg2sixdrr(states).to('cuda:0') # for rotation matrix
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)


class StateDependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states):
        #pdb.set_trace()
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp(-20, 2))
