import os
import numpy as np
import torch
import pdb
from transforms3d.euler import euler2mat, euler2quat
from transforms3d.quaternions import mat2quat

def deg2quat(deg_states):
    state = np.array([])
    states = []
    if len(deg_states.shape) == 1:
        for i in range(deg_states.shape[0]):
            hip_joint = [0, 0, deg_states[i].item()]
            quat = euler2quat(*hip_joint)
            state = np.concatenate((state, quat))
        states.append(state.tolist())
    
    else:
        for i in range(deg_states.shape[0]):
            state = np.array([])
            for j in range(deg_states.shape[1]):
                hip_joint = [0, 0, deg_states[i][j].item()]
                quat = euler2quat(*hip_joint)
                state = np.concatenate((state, quat))
            # pdb.set_trace()
            states.append(state.tolist())
    
    states = torch.Tensor(states)
    return states

class SerializedBuffer:

    def __init__(self, path, device):
        tmp = torch.load(path)
        self.buffer_size = self._n = tmp['state'].size(0)
        self.device = device

        self.states = tmp['state'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.dones = tmp['done'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)
        self.muscle_forces = tmp['muscle_force'].clone().to(self.device)
        self.excitations = tmp['excitation'].clone().to(self.device)
        self.activations = tmp['activation'].clone().to(self.device)
        self.rew_grfs = tmp['rew_grf'].clone().to(self.device)
        self.trunk_vels = tmp['trunk_vel'].clone().to(self.device)
        self.meta_costs = tmp['meta_cost'].clone().to(self.device)

        if 'quer_state' in tmp.keys():
            self.quer_states = tmp['quer_state'].to(self.device)
        if 'quer_next_state' in tmp.keys():
            self.quer_next_states = tmp['quer_next_state'].to(self.device)
        if 'obs_tx' in tmp.keys():
            self.obs_tx = tmp['obs_tx'].to(self.device)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            # self.quer_states[idxes], 
            self.trunk_vels[idxes], 
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )
    
    def sample_w_history(self, batch_size, history_length):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        idxes_matrix = np.zeros((batch_size, history_length), dtype=int)
        idxes_matrix[:, -1] = idxes
        for i in reversed(range(history_length - 1)): 
            idxes_matrix[:, i] = idxes_matrix[:, i + 1] - 1
        idxes_matrix[idxes_matrix < 0] = 0
        return (
            self.states[idxes_matrix],
            # self.quer_states[idxes], 
            self.trunk_vels[idxes], 
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )


class Buffer(SerializedBuffer): # Only sample, nothing to do with __init__ for collecting

    def __init__(self, buffer_size, state_shape, action_shape, muscle_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.muscle_forces = torch.empty(
            (buffer_size, muscle_shape), dtype=torch.float, device=device)
        self.excitations = torch.empty(
            (buffer_size, muscle_shape), dtype=torch.float, device=device)
        self.activations = torch.empty(
            (buffer_size, muscle_shape), dtype=torch.float, device=device)
        self.rew_grfs = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.trunk_vels = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.meta_costs = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.obs_tx = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state, muscle_force, excitation, activation, rew_grf, trunk_vel, meta_cost, obs_tx):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self.muscle_forces[self._p].copy_(torch.from_numpy(muscle_force))
        self.excitations[self._p].copy_(torch.from_numpy(excitation))
        self.activations[self._p].copy_(torch.from_numpy(activation))
        self.rew_grfs[self._p].copy_(torch.tensor(rew_grf))
        self.trunk_vels[self._p].copy_(torch.tensor(trunk_vel))
        self.meta_costs[self._p].copy_(torch.tensor(meta_cost))
        self.obs_tx[self._p].copy_(torch.tensor(obs_tx))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
            'muscle_force': self.muscle_forces.clone().cpu(),
            'excitation': self.excitations.clone().cpu(),
            'activation': self.activations.clone().cpu(),
            'rew_grf': self.rew_grfs.clone().cpu(),
            'trunk_vel': self.trunk_vels.clone().cpu(),
            'meta_cost': self.meta_costs.clone().cpu(),
            'obs_tx': self.obs_tx.clone().cpu(),
        }, path)


class QuerBuffer(Buffer): # Only sample, nothing to do with __init__

    def __init__(self, buffer_size, state_shape, action_shape, muscle_shape, device):
        super().__init__(buffer_size, state_shape, action_shape, muscle_shape, device)
        
        self.quer_states = torch.empty(
            (buffer_size, state_shape[0] * 4), dtype=torch.float, device=device)
        self.quer_next_states = torch.empty(
            (buffer_size, state_shape[0] * 4), dtype=torch.float, device=device)


    def copy(self, states, actions, rewards, dones, next_states, muscle_forces, excitations, activations, rew_grfs, trunk_vels, meta_costs):
        self.states = states.clone().cpu()
        self.actions = actions.clone().cpu()
        self.rewards = rewards.clone().cpu()
        self.dones = dones.clone().cpu()
        self.next_states = next_states.clone().cpu()
        self.muscle_forces = muscle_forces.clone().cpu()
        self.excitations = excitations.clone().cpu()
        self.activations = activations.clone().cpu()
        self.rew_grfs = rew_grfs.clone().cpu()
        self.trunk_vels = trunk_vels.clone().cpu()
        self.meta_costs = meta_costs.clone().cpu()

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'quer_state': self.quer_states.clone().cpu(),
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
            'quer_next_state': self.quer_next_states.clone().cpu(),
            'muscle_force': self.muscle_forces.clone().cpu(),
            'excitation': self.excitations.clone().cpu(),
            'activation': self.activations.clone().cpu(),
            'rew_grf': self.rew_grfs.clone().cpu(),
            'trunk_vel': self.trunk_vels.clone().cpu(),
            'meta_cost': self.meta_costs.clone().cpu(),
        }, path)


class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, muscle_shape, device, mix=1):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size
        self.clean = True

        self.states = torch.empty(
            (self.total_size, state_shape), dtype=torch.float, device=device)
        self.quer_states = torch.empty(
            (self.total_size, state_shape * 4), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, state_shape), dtype=torch.float, device=device)
        self.quer_next_states = torch.empty(
            (self.total_size, state_shape * 4), dtype=torch.float, device=device)
        # self.muscle_forces = torch.empty(
        #     (self.total_size, muscle_shape), dtype=torch.float, device=device)
        # self.excitations = torch.empty(
        #     (buffer_size, muscle_shape), dtype=torch.float, device=device)
        # self.activations = torch.empty(
        #     (buffer_size, muscle_shape), dtype=torch.float, device=device)
        self.trunk_vels = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.rew_grfs = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.rew_constrs = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, log_pi, next_state, trunk_vel, rew_grf, rew_constr):
        self.states[self._p].copy_(torch.from_numpy(state))
        # self.quer_states[self._p].copy_(deg2quat(state).squeeze())
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        # self.quer_next_states[self._p].copy_(deg2quat(next_state).squeeze())
        # self.muscle_forces[self._p].copy_(torch.from_numpy(muscle_force))
        # self.excitations[self._p].copy_(torch.from_numpy(excitation))
        # self.activations[self._p].copy_(torch.from_numpy(activation))
        self.trunk_vels[self._p] = float(trunk_vel)
        self.rew_grfs[self._p] = float(rew_grf)
        self.rew_constrs[self._p] = float(rew_constr)

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)
        self.clean = False


    def get(self):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.states[idxes],
            self.quer_states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes], 
            self.quer_next_states[idxes],
            # self.muscle_forces[idxes], 
            # self.excitations[idxes], 
            # self.activations[idxes], 
            self.trunk_vels[idxes],
            self.rew_grfs[idxes],
            self.rew_constrs[idxes],
        )
    
    def get_w_history(self, history_length):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        # idxes = slice(start, start + self.buffer_size)
        idxes = np.arange(start, start + self.buffer_size, dtype=int)
        idxes_matrix = np.zeros((self.buffer_size, history_length), dtype=int)
        idxes_matrix[:, -1] = idxes
        for i in reversed(range(history_length - 1)): 
            idxes_matrix[:, i] = idxes_matrix[:, i + 1] - 1
        idxes_matrix[idxes_matrix < 0] = 0
        return (
            self.states[idxes_matrix],
            self.quer_states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes_matrix], 
            self.quer_next_states[idxes],
            # self.muscle_forces[idxes], 
            # self.excitations[idxes], 
            # self.activations[idxes], 
            self.trunk_vels[idxes],
            self.rew_grfs[idxes],
            self.rew_constrs[idxes],
        )

    def sample(self, batch_size):
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            # self.quer_states[idxes],
            self.trunk_vels[idxes], 
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes], 
            self.quer_next_states[idxes],
            # self.muscle_forces[idxes], 
            # self.excitations[idxes], 
            # self.activations[idxes], 
        )
    
    def sample_w_history(self, batch_size, history_length):
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        idxes_matrix = np.zeros((batch_size, history_length), dtype=int)
        idxes_matrix[:, -1] = idxes
        for i in reversed(range(history_length - 1)): 
            idxes_matrix[:, i] = idxes_matrix[:, i + 1] - 1
        idxes_matrix[idxes_matrix < 0] = 0
        return (
            self.states[idxes_matrix],
            # self.quer_states[idxes],
            self.trunk_vels[idxes], 
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes_matrix], 
            self.quer_next_states[idxes],
            # self.muscle_forces[idxes], 
            # self.excitations[idxes], 
            # self.activations[idxes], 
        )
    ###################################################################################################################
    def copy2sum(self, buffer):
        # assert states.shape[0] == actions.shape[0]
        # assert states.shape[0] == rewards.shape[0]
        # assert states.shape[0] == dones.shape[0]
        # assert states.shape[0] == log_pis.shape[0]
        # assert states.shape[0] == next_states.shape[0]
        # assert states.shape[0] == trunk_vels.shape[0]
        # assert states.shape[0] == rew_grfs.shape[0]
        # assert states.shape[0] == rew_constrs.shape[0]


        self.states[self._p:buffer.states.shape[0]].copy_(buffer.states)
        # self.quer_states[self._p].copy_(deg2quat(state).squeeze())
        self.actions[self._p:self._p + buffer._n].copy_(buffer.actions)
        self.rewards[self._p:self._p + buffer._n] = buffer.rewards
        self.dones[self._p:self._p + buffer._n] = buffer.dones
        self.log_pis[self._p:self._p + buffer._n] = buffer.log_pis
        self.next_states[self._p:self._p + buffer._n].copy_(buffer.next_states)
        # self.quer_next_states[self._p].copy_(deg2quat(next_state).squeeze())
        # self.muscle_forces[self._p].copy_(torch.from_numpy(muscle_force))
        # self.excitations[self._p].copy_(torch.from_numpy(excitation))
        # self.activations[self._p].copy_(torch.from_numpy(activation))
        self.trunk_vels[self._p:self._p + buffer._n] = buffer.trunk_vels
        self.rew_grfs[self._p:self._p + buffer._n] = buffer.rew_grfs
        self.rew_constrs[self._p:self._p + buffer._n] = buffer.rew_constrs

        self._p = (self._p + buffer._n) % self.total_size
        self._n = min(self._n + buffer._n, self.total_size)
    

    def sample_all(self, batch_size):
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            idxes, 
            self.states[idxes],
            self.quer_states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes], 
            self.quer_next_states[idxes],
            # self.muscle_forces[idxes], 
            # self.excitations[idxes], 
            # self.activations[idxes], 
            self.trunk_vels[idxes],
            self.rew_grfs[idxes],
            self.rew_constrs[idxes],
        )
