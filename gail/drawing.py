import os
import sys

import numpy as np
import wandb
import matplotlib.pyplot as plt
import argparse
import torch
import pdb

from matplotlib.backends.backend_agg import FigureCanvasAgg

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from gail_demo.buffer import SerializedBuffer
from expert_sim import simulate
from dtw import *
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

def mplfig_to_npimage(fig):
    """ Converts a matplotlib figure to a RGB frame after updating the canvas"""
    #  only the Agg backend now supports the tostring_rgb function
    
    canvas = FigureCanvasAgg(fig)
    canvas.draw() # update/draw the elements

    # get the width and the height to resize the matrix
    l,b,w,h = canvas.figure.bbox.bounds
    w, h = int(w), int(h)

    #  exports the canvas to a string buffer and then to a numpy nd.array
    buf = canvas.tostring_rgb()
    image= np.frombuffer(buf,dtype=np.uint8)
    return image.reshape(h,w,3)


def comp_states_nextstates(args):

    buffer = SerializedBuffer(path=args.buffer, device=torch.device("cuda" if args.cuda else "cpu"))

    steps = np.linspace(0,100,100, dtype=int)
    #pdb.set_trace()
    
    for idx in range(buffer.states[0].shape[0]):
        i = int(idx / 25) #fig_{i}
        if idx % 25 == 0:
            locals()[f'fig_{i}'], locals()[f'axs_{i}'] = plt.subplots(5,5)
            locals()[f'axs_{i}'] = np.array(locals()[f'axs_{i}'])

            locals()[f'fig_{i}'].suptitle("States vs. Next_states")

        x = int((idx % 25) / 5)
        y = (idx % 25) % 5
        #pdb.set_trace()
        ax = locals()[f'axs_{i}'][x][y]
        ax.plot(steps, np.array(buffer.states[steps, :][:, idx].cpu()), label="states")
        ax.plot(steps, np.array(buffer.next_states[steps, :][:, idx].cpu()), label="next_states")
        #ax.set_yticks(np.arange(-4, 4, 0.5))
        ax.legend(loc='upper right')
        ax.set_xlabel("steps")

        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        #plt.tight_layout()

    #img = mplfig_to_npimage(fig)

    #wandb_img = wandb.Image(img)
    # wandb.log({"analysis": wandb_img})

    #manager = plt.get_current_fig_manager()
    #manager.resize(*manager.window.maxsize())
    #manager.full_screen_toggle()

    plt.show()


def freq_analysis(series, window, sample_rate, n=3):
    seriesf = fft(series)
    xf = fftfreq(window, 1.0 / sample_rate)
    seriesf_pos = 2.0/window * np.abs(seriesf[0:window // 2])
    xf_pos = xf[:window // 2]
    dominant_omega, peaks = find_dominant_freq(xf_pos, seriesf_pos, n)

    return xf_pos, seriesf_pos, dominant_omega, peaks

def find_dominant_freq(xf, seriesf, n=3):
    peaks, _ = find_peaks(seriesf)
    dominamt_num = []
    dominant_tmp = seriesf[peaks]
    
    for i in range(n): 
        # dominamt_num.append(peaks[i])
        
        # If want to use the energy criteria
        if len(dominant_tmp.tolist()) != 0:
            dominamt_num.append(peaks[dominant_tmp.argmax()])
            dominant_tmp[dominant_tmp.argmax()] = -np.inf
        else:
            dominamt_num.append(1)
    
    return xf[dominamt_num], dominamt_num





class Dof_store:
    def __init__(self, max_step=100000, device="cuda" if torch.cuda.is_available else "cpu"):
        self.states_trained = torch.zeros((max_step, 18), device=device)
        self.cur_step = 0
        self._p = 0
        self.max_step = max_step
        self.device = device
    def append(self, state, obs_tx):     # Append data from trained agent
        state[1] = obs_tx
        self.states_trained[self._p].copy_(torch.from_numpy(state))
        self._p = (self._p + 1) % self.max_step
    def clear(self):
        self.states_trained = torch.zeros((self.max_step, 18), device=self.device)
        self._p = 0
    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states_trained.clone().cpu(), 
        }, path)






class Carrier:

    def __init__(self, agent_env_id, expert_env_id, expert_analysis=False, max_step=100000, device="cuda" if torch.cuda.is_available else "cpu"): # Load stored data for normal and patho, while initialize the to be trained

        if "shorthamstring" in agent_env_id:
            pass
        # self.buffer_normal = SerializedBuffer(path="buffers/sconewalk_origin_motored_normal_h0914-v3/v1/Adjusted_Same70Init_dof_size1000000_for_shorthamstring_new_freq1.11_100hz.pth", device=device)

        # Short hamstring
        if agent_env_id == 'sconewalk_origin_motored_shorthamstring_h0914-v3': 
            self.buffer_normal = SerializedBuffer(path=os.path.expanduser("~") + "/MA/codes/gail/buffers/sconewalk_origin_motored_normal_h0914-v3/v2/Adjusted_Same70Init_dof_size888000_for_shorthamstring_new_freq1.11_100hz.pth", device=device)
            self.buffer_patho = SerializedBuffer(path=os.path.expanduser("~") + "/MA/codes/gail/buffers/sconewalk_origin_motored_shorthamstring_h0914-v3/v1/Same70Init_dof_size1000000_std0.0_prand1.0_100hz.pth", device=device)
        
        # Plantar weakness
        # self.buffer_normal = SerializedBuffer(path="buffers/sconewalk_origin_motored_normal_h0914-v3/v1/Adjusted_Same70Init_dof_size1000000_for_hamstringweakness_new_freq1.43_100hz.pth", device=device)
        # self.buffer_patho = SerializedBuffer(path="buffers/sconewalk_origin_motored_hamstringweakness_h0914-v3/v1/Same70Init_dof_size1000000_std0.0_prand1.0_100hz.pth", device=device)
        # Hamstring weakness
        if agent_env_id == 'sconewalk_origin_motored_hamstringweakness_new_h0914-v3': 
            self.buffer_normal = SerializedBuffer(path=os.path.expanduser("~") + "/MA/codes/gail/buffers/sconewalk_origin_motored_normal_h0914-v3/v2/Adjusted_Same70Init_dof_size1144000_for_hamstringweakness_new_freq1.43_100hz.pth", device=device)
            self.buffer_patho = SerializedBuffer(path=os.path.expanduser("~") + "/MA/codes/gail/buffers/sconewalk_origin_motored_hamstringweakness_h0914-v3/v1/Same70Init_dof_size1000000_std0.0_prand1.0_100hz.pth", device=device)
        # Iliopsoas weakness /home/stud_zhang/MA/codes/gail/
        if agent_env_id == 'sconewalk_origin_motored_iliopsoasweakness_h0914-v3': 
            self.buffer_normal = SerializedBuffer(path=os.path.expanduser("~") + "/MA/codes/gail/buffers/sconewalk_origin_motored_normal_h0914-v3/v2/Adjusted_Same70Init_dof_size888000_for_iliopsoasweakness_new_freq1.11_100hz.pth", device=device)
            self.buffer_patho = SerializedBuffer(path=os.path.expanduser("~") + "/MA/codes/gail/buffers/sconewalk_origin_motored_iliopsoasweakness_h0914-v3/v1/Same70Init_dof_size1000000_std0.0_prand1.0_100hz.pth", device=device)
        
        # pdb.set_trace()
        self.states_trained = torch.zeros((max_step, self.buffer_normal.states.shape[1]), device=device)
        self.actions_trained = torch.zeros((max_step, self.buffer_normal.actions.shape[1]), device=device)
        self.rewards_trained = torch.zeros((max_step, self.buffer_normal.rewards.shape[1]), device=device)
        self.muscle_force_trained = torch.zeros((max_step, self.buffer_normal.muscle_forces.shape[1]), device=device)
        self.excitation_trained = torch.zeros((max_step, self.buffer_normal.excitations.shape[1]), device=device)
        self.activation_trained = torch.zeros((max_step, self.buffer_normal.activations.shape[1]), device=device)
        self.rew_grf_trained = torch.zeros((max_step, self.buffer_normal.rew_grfs.shape[1]), device=device)
        self.trunk_vel_trained = torch.zeros((max_step, self.buffer_normal.trunk_vels.shape[1]), device=device)
        self.meta_cost_trained = torch.zeros((max_step, self.buffer_normal.meta_costs.shape[1]), device=device)

        # self.states_expert = torch.empty((1, self.buffer_normal.states.shape[1]), device=device)
        # self.muscle_force_expert = torch.empty((1, self.buffer_normal.muscle_forces.shape[1]), device=device)
        # self.excitation_expert = torch.empty((1, self.buffer_normal.excitations.shape[1]), device=device)
        # self.activation_expert = torch.empty((1, self.buffer_normal.activations.shape[1]), device=device)


        self.initial = True
        self.grids = np.linspace(0,500,100, dtype=int)

        self.device = device

        self.i = 0 # idx for global variant plot counts
        self.cur_step = 0
        self._p = 0
        self.max_step = max_step

        self.dissimilarity_list = []
        self.objective_list = []
        self.mse_list = []
        self.mae_list = []

        if expert_analysis:
            # simulate(expert_env_id)
            pass
    
    def append_agent(self, state, action, reward, muscle_force, excitation, activation, rew_grf, trunk_vel, meta_cost):     # Append data from trained agent
        self.states_trained[self._p].copy_(torch.from_numpy(state))
        self.actions_trained[self._p].copy_(torch.from_numpy(action))
        self.rewards_trained[self._p].copy_(torch.tensor(reward))
        self.muscle_force_trained[self._p].copy_(torch.from_numpy(muscle_force))
        self.excitation_trained[self._p].copy_(torch.from_numpy(excitation))
        self.activation_trained[self._p].copy_(torch.from_numpy(activation))
        self.rew_grf_trained[self._p].copy_(torch.tensor(rew_grf))
        self.trunk_vel_trained[self._p].copy_(torch.tensor(trunk_vel))
        self.meta_cost_trained[self._p].copy_(torch.tensor(meta_cost))

        self._p = (self._p + 1) % self.max_step

        
    
    def clear_agent(self):
        self.states_trained = torch.zeros((self.max_step, self.buffer_normal.states.shape[1]), device=self.device)
        self.actions_trained = torch.zeros((self.max_step, self.buffer_normal.actions.shape[1]), device=self.device)
        self.rewards_trained = torch.zeros((self.max_step, self.buffer_normal.rewards.shape[1]), device=self.device)
        self.muscle_force_trained = torch.zeros((self.max_step, self.buffer_normal.muscle_forces.shape[1]), device=self.device)
        self.excitation_trained = torch.zeros((self.max_step, self.buffer_normal.excitations.shape[1]), device=self.device)
        self.activation_trained = torch.zeros((self.max_step, self.buffer_normal.activations.shape[1]), device=self.device)
        self.rew_grf_trained = torch.zeros((self.max_step, self.buffer_normal.rew_grfs.shape[1]), device=self.device)
        self.trunk_vel_trained = torch.zeros((self.max_step, self.buffer_normal.trunk_vels.shape[1]), device=self.device)
        self.meta_cost_trained = torch.zeros((self.max_step, self.buffer_normal.meta_costs.shape[1]), device=self.device)
        self._p = 0

        self.dissimilarity_list = []
        self.objective_list = []
        self.mse_list = []
        self.mae_list = []



    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states_trained.clone().cpu(),
            'action': self.actions_trained.clone().cpu(),
            'reward': self.rewards_trained.clone().cpu(),
            'muscle_force': self.muscle_force_trained.clone().cpu(),
            'excitation': self.excitation_trained.clone().cpu(),
            'activation': self.activation_trained.clone().cpu(),
            'rew_grf': self.rew_grf_trained.clone().cpu(),
            'trunk_vel': self.trunk_vel_trained.clone().cpu(),
            'meta_cost': self.meta_cost_trained.clone().cpu(),
            # 'dissimilarity': torch.Tensor(self.dissimilarity_list), 
            # 'objective': torch.Tensor(self.objective_list), 
            # 'mse': torch.Tensor(self.mse_list), 
            # 'mae': torch.Tensor(self.mae_list)
        }, path)


    def calc_dissimilar(self, init_step, init_range):
        if init_range > self._p:
            init_range = self._p
        self.pi_rew_grf = self.rew_grf_trained[:init_range][:]
        self.ex_rew_grf = self.buffer_normal.rew_grfs[init_step:init_range + init_step][:]
        self.ex_rew_grf_mean = self.buffer_normal.rew_grfs.mean()
        self.ex_rew_grf_std = self.buffer_normal.rew_grfs.std()

        self.ex_dof_pos = self.buffer_normal.states[init_step:init_range + init_step, :self.buffer_normal.states.shape[1] // 2]
        self.ex_dof_vel = self.buffer_normal.states[init_step:init_range + init_step, self.buffer_normal.states.shape[1] // 2:]
        self.pi_dof_pos =  self.states_trained[:init_range, :self.states_trained.shape[1] // 2]
        self.pi_dof_vel =  self.states_trained[:init_range, self.states_trained.shape[1] // 2:]
        self.ex_dof_pos_mean = self.buffer_normal.states[:, :self.buffer_normal.states.shape[1] // 2].mean(axis=0)
        self.ex_dof_pos_std = self.buffer_normal.states[:, :self.buffer_normal.states.shape[1] // 2].std(axis=0)
        self.ex_dof_vel_mean = self.buffer_normal.states[:, self.buffer_normal.states.shape[1] // 2:].mean(axis=0)
        self.ex_dof_vel_std = self.buffer_normal.states[:, self.buffer_normal.states.shape[1] // 2:].std(axis=0)
        # pdb.set_trace()
        # self.norm_ex_dof_pos = self.normalize2d(self.ex_dof_pos, self.ex_dof_pos_mean, self.ex_dof_pos_std)
        # self.norm_ex_dof_vel = self.normalize2d(self.ex_dof_vel, self.ex_dof_vel_mean, self.ex_dof_vel_std)
        # self.norm_pi_dof_pos = self.normalize2d(self.pi_dof_pos, self.ex_dof_pos_mean, self.ex_dof_pos_std)
        # self.norm_pi_dof_vel = self.normalize2d(self.pi_dof_vel, self.ex_dof_vel_mean, self.ex_dof_vel_std)

        # norm_dissimilarity = (((self.norm_pi_rew_grf - self.norm_ex_rew_grf)**2).sum() + ((self.norm_pi_dof_pos - self.norm_ex_dof_pos)**2).sum() + 
        #                  ((self.norm_pi_dof_vel - self.norm_ex_dof_vel)**2).sum() + (np.array(self.pi_trunk_vel).mean() - np.array(self.ex_trunk_vel).mean()) +
        #                  (np.array(self.pi_meta_cost).mean() - np.array(self.ex_meta_cost).mean()) )
        # mse_loss = (((self.pi_dof_pos - self.ex_dof_pos)**2).sum() + ((self.pi_dof_pos - self.ex_dof_vel)**2).sum()) / init_range
        # mae_loss = (((self.pi_dof_pos - self.ex_dof_pos).absolute()).sum() + ((self.pi_dof_pos - self.ex_dof_vel).absolute()).sum()) / init_range

        # dissimilarity = ((((self.pi_rew_grf - self.ex_rew_grf_mean) / (self.ex_rew_grf_std + 1e-6))**2).sum() + (((self.pi_dof_pos - self.ex_dof_pos_mean) / (self.ex_dof_pos_std + 1e-6))**2).sum() + 
        #                  (((self.pi_dof_vel - self.ex_dof_vel) / (self.ex_dof_vel_std + 1e-6))**2).sum() + ((self.trunk_vel_trained[:init_range] - self.buffer_normal.trunk_vels[init_step:init_range + init_step])**2).sum() +
        #                  (self.meta_cost_trained[:init_range] - self.buffer_normal.meta_costs[init_step:init_range + init_step]).mean() ) / init_range
        # ex_dof_pos_normalized = (self.ex_dof_pos - self.ex_dof_pos_mean) / (self.ex_dof_pos_std + 1e-6)
        mse_loss_pos = (((self.pi_dof_pos - self.ex_dof_pos)**2).sum()) / init_range
        mae_loss_pos = (((self.pi_dof_pos - self.ex_dof_pos).absolute()).sum()) / init_range

        mse_loss_vel = (((self.pi_dof_vel - self.ex_dof_vel)**2).sum()) / init_range
        mae_loss_vel = (((self.pi_dof_vel - self.ex_dof_vel).absolute()).sum()) / init_range

        # dissimilarity = ((((self.pi_rew_grf - self.ex_rew_grf_mean) / (self.ex_rew_grf_std + 1e-6))**2).sum() + (((self.pi_dof_pos - self.ex_dof_pos_mean) / (self.ex_dof_pos_std + 1e-6))**2).sum() + 
        #                  ((self.trunk_vel_trained[:init_range] - self.buffer_normal.trunk_vels[init_step:init_range + init_step])**2).sum() +
        #                  (self.meta_cost_trained[:init_range] - self.buffer_normal.meta_costs[init_step:init_range + init_step]).mean() ) / init_range
        
        # objective = dissimilarity - self._p // 1000
        dissimilarity = objective = 0
        

        dtw_normalizeddistance_pos = 0
        dtw_normalizeddistance_vel = 0
        # query = ((self.pi_dof_pos - self.ex_dof_pos_mean) / (self.ex_dof_pos_std + 1e-6)).cpu().numpy()
        # template = ((self.ex_dof_pos - self.ex_dof_pos_mean) / (self.ex_dof_pos_std + 1e-6)).cpu().numpy()
        # query = self.pi_dof_pos.cpu().numpy()
        # template = self.ex_dof_pos.cpu().numpy()
        # for k in range(query.shape[1]): 
        #     alignment = dtw(query[:, k], template[:, k], keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c"))
        #     dtw_normalizeddistance += alignment.normalizedDistance
        #     dtw_distance += alignment.distance

        dtw_normalizeddistance_pos, _, _, _, _, _, _ = self.align_dtw(init_step, init_range, use_open_end=True)
        # dtw_normalizeddistance_vel, _, _, _, _, _, _ = self.align_dtw(init_step, init_range, use_vel=True)
        
        return dissimilarity, objective, mse_loss_pos, mae_loss_pos, mse_loss_vel, mae_loss_vel, dtw_normalizeddistance_pos, dtw_normalizeddistance_vel
    


    def align_dtw(self, init_step, init_range, use_vel=False, use_open_end=False, sachochiba=None):
            switch = False
            if init_range > self._p:
                init_range = self._p
            # Setteled: using expert as query and agent as reference
            if not use_vel:
                query = self.buffer_normal.states[init_step:init_range + init_step, :self.buffer_normal.states.shape[1] // 2].cpu().numpy()
                reference = self.states_trained[:init_range, :self.states_trained.shape[1] // 2].cpu().numpy()
            else:
                query = self.buffer_normal.states[init_step:init_range + init_step, self.buffer_normal.states.shape[1] // 2:].cpu().numpy()
                reference = self.states_trained[:init_range, self.states_trained.shape[1] // 2:].cpu().numpy()
            # _, _, dominant_omega_ref, peaks_ref = freq_analysis(reference_0[3], reference_0.shape[0], sample_rate=0.01, n=1)
            # _, _, dominant_omega_que, peaks_que = freq_analysis(query_0[3], query_0.shape[0], sample_rate=0.01, n=1)
            # pdb.set_trace()
            # reference = reference_0
            # query = query_0
            # if float(peaks_ref[0]) < float(peaks_que[0]):
            #     reference = query_0
            #     query = reference_0
            #     switch = True
            if sachochiba is not None:
                w = int(sachochiba * reference.shape[0])
            n = reference.shape[0]
            m = query.shape[0]
            if sachochiba is None:
                DTW = np.zeros([n+1, m+1])
                DTW[0,:] = np.inf
                DTW[:,0] = np.inf
                DTW[0,0] = 0
            else:
                DTW = np.full((n+1, m+1), np.inf)
                for i in range(1, n+1):
                    DTW[i, max(1, i-w):min(m+1, i+w+1)] = 0
                DTW[0,0] = 0

            for i in range(n):
                for j in range(m):
                    if (sachochiba is None or (max(0, i-w) <= j <= min(m, i+w))): 
                        cost = np.linalg.norm(query[i,:] - reference[j,:])
                        DTW[i+1,j+1] = cost + min([DTW[i, j+1], DTW[i+1, j], DTW[i, j] ])

            DTW = DTW[1:,1:]
            i = DTW.shape[0] - 1
            j = DTW.shape[1] - 1
            if use_open_end:
                j = np.nanargmin(DTW[-1,:])
            matches = []
            mappings_series_1 = [list() for v in range(DTW.shape[0])]
            mappings_series_2 = [list() for v in range(DTW.shape[1])]
            while i > 0 or j > 0:
                matches.append((i, j))
                mappings_series_1[i].append(j)
                mappings_series_2[j].append(i)
                move_diag = DTW[i - 1, j - 1] if i > 0 and j > 0 else np.inf
                move_up = DTW[i - 1, j] if i > 0 else np.inf
                move_left = DTW[i, j - 1] if j > 0 else np.inf
                # move_diag *= 1.01 # sakoe-chiba band
                move = np.argmin([move_diag * 1.0, move_up, move_left]) 
                if move == 0:
                    i -= 1
                    j -= 1
                elif move == 1:
                    i -= 1
                else:
                    j -= 1
            matches.append((0, 0))
            mappings_series_1[0].append(0)
            mappings_series_2[0].append(0)
            matches.reverse()
            for mp in mappings_series_1:
                mp.reverse()
            for mp in mappings_series_2:
                mp.reverse()
            
            return DTW[-1, -1]/init_range, DTW, mappings_series_1, mappings_series_2, matches, query, reference
    
    
    
    
    def upload_plot_dtw(self, env_name, step, ylab_list, query, reference, matches, use_vel=False):
        # DTW metrics
        fig_dtw_1, axs_dtw_1 = plt.subplots(3, 1, figsize=(10, (2 * 2) * 3))
        fig_dtw_2, axs_dtw_2 = plt.subplots(3, 1, figsize=(10, (2 * 2) * 3))
        fig_dtw_3, axs_dtw_3 = plt.subplots(3, 1, figsize=(10, (2 * 2) * 3))
        fig_dtw_1.suptitle(f'Dofs: dtw_{step}_steps, from 0s')
        fig_dtw_2.suptitle(f'Dofs: dtw_{step}_steps, from 0s')
        fig_dtw_3.suptitle(f'Dofs: dtw_{step}_steps, from 0s')
        
        if query.shape[0] > 2000:
            query = query[0:2000, :]
            reference = reference[0:2000, :]
            for k, v in enumerate(matches):
                if v[0] >= 2000 or v[1] >= 2000:
                    matches = matches[:k-1]

        for k in range(query.shape[1]): 
            if k / 3 == 0:
                axs_dtw = axs_dtw_1
            elif k / 3 == 1:
                axs_dtw = axs_dtw_2
            elif k / 3 == 2:
                axs_dtw = axs_dtw_3

            # alignment = dtw(query[:, k], reference[:, k], keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c"))
            # wandb.log({f'final_example_dtw_distance_{ylab_list[k]}': alignment.distance})
            # dtw_normalizeddistance_sum += alignment.normalizedDistance
            # dtw_distance_sum += alignment.distance
            self.dtw_plot(axs_dtw[k%3], query[:, k], reference[:, k], matches, ylab=ylab_list[k])

        fig_dtw_1.set_size_inches(16, 10)
        fig_dtw_2.set_size_inches(16, 10)
        fig_dtw_3.set_size_inches(16, 10)
        # plt.show()
        # pdb.set_trace()
        if not use_vel:
            wandb_img = wandb.Image(fig_dtw_1)
            wandb.log({f'dtw_pos_{env_name}': wandb_img})
            wandb_img = wandb.Image(fig_dtw_2)
            wandb.log({f'dtw_pos_{env_name}': wandb_img})
            wandb_img = wandb.Image(fig_dtw_3)
            wandb.log({f'dtw_pos_{env_name}': wandb_img})
        else: 
            wandb_img = wandb.Image(fig_dtw_1)
            wandb.log({f'dtw_vel_{env_name}': wandb_img})
            wandb_img = wandb.Image(fig_dtw_2)
            wandb.log({f'dtw_vel_{env_name}': wandb_img})
            wandb_img = wandb.Image(fig_dtw_3)
            wandb.log({f'dtw_vel_{env_name}': wandb_img})
        
        plt.close(fig_dtw_1)
        plt.close(fig_dtw_2)
        plt.close(fig_dtw_3)
    

    def upload_plot_dtw_indivi(self, env_name, step, ylab_list, init_step, init_range):
        # DTW metrics
        fig_dtw_1, axs_dtw_1 = plt.subplots(3, 1, figsize=(10, (2 * 2) * 3))
        fig_dtw_2, axs_dtw_2 = plt.subplots(3, 1, figsize=(10, (2 * 2) * 3))
        fig_dtw_3, axs_dtw_3 = plt.subplots(3, 1, figsize=(10, (2 * 2) * 3))
        fig_dtw_1.suptitle(f'Dofs: dtw_{step}_steps, from 0s')
        fig_dtw_2.suptitle(f'Dofs: dtw_{step}_steps, from 0s')
        fig_dtw_3.suptitle(f'Dofs: dtw_{step}_steps, from 0s')
        
        if init_range > self._p:
            init_range = self._p
        
        for k in range(self.buffer_normal.states.shape[1] // 2): 
            if k / 3 == 0:
                axs_dtw = axs_dtw_1
            elif k / 3 == 1:
                axs_dtw = axs_dtw_2
            elif k / 3 == 2:
                axs_dtw = axs_dtw_3

            # reference = self.buffer_normal.states[init_step:init_range + init_step, k].cpu().numpy()
            # query = self.states_trained[:init_range, k].cpu().numpy()
            # switch query and reference

            query = self.buffer_normal.states[init_step:init_range + init_step, k].cpu().numpy()
            reference = self.states_trained[:init_range, k].cpu().numpy()
            # reference = query_0
            # query = reference_0
            # switch = False
            # _, _, dominant_omega_ref, peaks_ref = freq_analysis(reference_0, init_range, sample_rate=0.01, n=1)
            # _, _, dominant_omega_que, peaks_que = freq_analysis(query_0, init_range, sample_rate=0.01, n=1)
            # if float(peaks_ref[0]) < float(peaks_que[0]):
            #     reference = query_0
            #     query = reference_0
            #     switch = True
                    

            n = reference.shape[0]
            m = query.shape[0]
            DTW = np.zeros([n+1, m+1])
            DTW[0,:] = np.inf
            DTW[:,0] = np.inf
            DTW[0,0] = 0
            for i in range(n):
                for j in range(m):
                    cost = np.linalg.norm(query[i] - reference[j])
                    DTW[i+1,j+1] = cost + min([DTW[i, j+1], DTW[i+1, j], DTW[i, j] ])

            DTW = DTW[1:,1:]
            i = DTW.shape[0] - 1
            j = DTW.shape[1] - 1
            matches = []
            mappings_series_1 = [list() for v in range(DTW.shape[0])]
            mappings_series_2 = [list() for v in range(DTW.shape[1])]
            while i > 0 or j > 0:
                matches.append((i, j))
                mappings_series_1[i].append(j)
                mappings_series_2[j].append(i)
                move_diag = DTW[i - 1, j - 1] if i > 0 and j > 0 else np.inf
                move_up = DTW[i - 1, j] if i > 0 else np.inf
                move_left = DTW[i, j - 1] if j > 0 else np.inf
                # move_diag *= 1.01 # sakoe-chiba band
                move = np.argmin([move_diag * 1.01, move_up, move_left]) 
                if move == 0:
                    i -= 1
                    j -= 1
                elif move == 1:
                    i -= 1
                else:
                    j -= 1
            matches.append((0, 0))
            mappings_series_1[0].append(0)
            mappings_series_2[0].append(0)
            matches.reverse()
            # for mp in mappings_series_1:
            #     mp.reverse()
            # for mp in mappings_series_2:
            #     mp.reverse()

            if query.shape[0] > 4000:
                query = query[0:4000]
                reference = reference[0:4000]
                for k_v, v in enumerate(matches):
                    if v[0] >= 4000 or v[1] >= 4000:
                        matches = matches[:k_v-1]

            # alignment = dtw(query[:, k], reference[:, k], keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c"))
            # wandb.log({f'final_example_dtw_distance_{ylab_list[k]}': alignment.distance})
            # dtw_normalizeddistance_sum += alignment.normalizedDistance
            # dtw_distance_sum += alignment.distance
            # print('k:', k)
            # print('query:', query.shape)
            # print('reference:', reference.shape)
            self.dtw_plot(axs_dtw[k%3], reference, query, matches, ylab=ylab_list[k])

        fig_dtw_1.set_size_inches(16, 10)
        fig_dtw_2.set_size_inches(16, 10)
        fig_dtw_3.set_size_inches(16, 10)
        # plt.show()
        # pdb.set_trace()
        wandb_img = wandb.Image(fig_dtw_1)
        wandb.log({f'dtw_plot_{env_name}': wandb_img})
        wandb_img = wandb.Image(fig_dtw_2)
        wandb.log({f'dtw_plot_{env_name}': wandb_img})
        wandb_img = wandb.Image(fig_dtw_3)
        wandb.log({f'dtw_plot_{env_name}': wandb_img})
        plt.close(fig_dtw_1)
        plt.close(fig_dtw_2)
        plt.close(fig_dtw_3)

    
        

    
    def dtw_plot(self, ax, xts, yts, match_indices,
                  offset=2,
                  ts_type="l",                  
                  match_col="gray",
                  xlab="Index",
                  ylab="Query value",
                  **kwargs):
        
        from matplotlib import collections  as mc

        # if xts is None or yts is None:
        #     try:
        #         xts = d.query
        #         yts = d.reference
        #     except:
        #         raise ValueError("Original timeseries are required")
        

        offset = -offset

        xtimes = numpy.arange(len(xts))
        ytimes = numpy.arange(len(yts))
        
        # ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        
        ax.plot(xtimes, numpy.array(xts), color='k', **kwargs)
        ax.plot(ytimes, numpy.array(yts) - offset, **kwargs)      # Plot with offset applied

        if offset != 0:
            # Create an offset axis
            ax2 = ax.twinx()
            ax2.tick_params('y', colors='b')
            ql, qh = ax.get_ylim()
            ax2.set_ylim(ql + offset, qh + offset)

        # https://stackoverflow.com/questions/21352580/matplotlib-plotting-numerous-disconnected-line-segments-with-different-colors
        # if match_indices is None:
        #     idx = numpy.linspace(0, len(d.index1) - 1)
        # elif not hasattr(match_indices, "__len__"):
        #     idx = numpy.linspace(0, len(d.index1) - 1, num=match_indices)
        # else:
        #     idx = match_indices
        # idx = numpy.array(idx).astype(int)


        col = []
        for i in match_indices:
            col.append([(i[1], xts[i[1]]),
                        (i[0], -offset + yts[i[0]])])
        lc = mc.LineCollection(col, linewidths=1, linestyles=":", colors=match_col)
        ax.add_collection(lc)

    def calc_plot_action(self, env_name, step):
        action_label = ['Hip_r', 'Knee_r', 'Hip_l', 'Knee_l']
        actions_input = 50 * np.tanh(self.actions_trained[:self._p].cpu().numpy())
        fig, axs = plt.subplots(4, 1, figsize=(10, (2 * 2) * 3))
        fig.suptitle(f'Actions_example: {step}_steps, from 0s')

        if actions_input.shape[0] > 1500:
            fig.suptitle(f'Actions_example: {step}_steps, from 10s')
            actions_input = actions_input[1000:1200, :]

        for k in range(actions_input.shape[1]): 
            action_input = actions_input[:, k]
            axs[k].plot(np.arange(len(action_input)), action_input)
            axs[k].set_ylabel(action_label[k])
        # pdb.set_trace()

        fig.set_size_inches(16, 10)
        wandb_img = wandb.Image(fig)
        wandb.log({f'Actions_example_{env_name}': wandb_img})
        plt.close(fig)

    # For plotting the figures used in thesis

    def save_plot_dtw_for_thesis(self, env_name, step, ylab_list, query, reference, matches, use_vel=False, half=False, path=None, timestep=None):
        # DTW metrics
        
        # fig_dtw_3, axs_dtw_3 = plt.subplots(3, 1, figsize=(10, (2 * 2) * 3))
        # fig_dtw_1.suptitle(f'Dofs: dtw_{step}_steps, from 0s')
        # fig_dtw_2.suptitle(f'Dofs: dtw_{step}_steps, from 0s')
        # fig_dtw_3.suptitle(f'Dofs: dtw_{step}_steps, from 0s')
        
        if query.shape[0] > 2000:
            query = query[0:2000, :]
            reference = reference[0:2000, :]
            for k, v in enumerate(matches):
                if v[0] >= 2000 or v[1] >= 2000:
                    matches = matches[:k-1]

        ylab_list = ['Pelvis Tilt (rad)', 
                  'Pelvis Height (m)', 
                  'Right Hip (rad)', 
                  'Right Knee (rad)', 
                  'Right Ankle (rad)', 
                  'Left Hip (rad)', 
                  'Left Knee (rad)', 
                  'Left Ankle (rad)',]
        if half:
            fig_dtw_1, axs_dtw_1 = plt.subplots(4, 1)
            fig_dtw_2, axs_dtw_2 = plt.subplots(4, 1)
            for k, dof in enumerate([0, 2, 3, 4, 5, 6, 7, 8]):
                if k // 4 == 0:
                    axs_dtw = axs_dtw_1
                    fig_dtw = fig_dtw_1
                elif k // 4 == 1:
                    axs_dtw = axs_dtw_2
                    fig_dtw = fig_dtw_2
                # elif k / 3 == 2:
                #     axs_dtw = axs_dtw_3

                # alignment = dtw(query[:, k], reference[:, k], keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c"))
                # wandb.log({f'final_example_dtw_distance_{ylab_list[k]}': alignment.distance})
                # dtw_normalizeddistance_sum += alignment.normalizedDistance
                # dtw_distance_sum += alignment.distance
                self.dtw_plot_for_thesis(axs_dtw[k%4], query[:, dof], reference[:, dof], matches, ylab=ylab_list[k], k=k, fig=fig_dtw)

                
                if k == 3 or k == 7:
                    axs_dtw[k%4].set_xlabel('Time Step (environment)')

            fig_dtw_1.set_size_inches(10, 8)
            fig_dtw_2.set_size_inches(10, 8)
            fig_dtw_1.subplots_adjust(top=0.95, bottom=0.07, left=0.06, right=0.93)
            fig_dtw_2.subplots_adjust(top=0.95, bottom=0.07, left=0.06, right=0.93)
            plt.close(fig_dtw_1)
            plt.close(fig_dtw_2)
        else:
            fig_dtw, axs_dtw = plt.subplots(8, 1)
            for k, dof in enumerate([0, 2, 3, 4, 5, 6, 7, 8]):
                # alignment = dtw(query[:, k], reference[:, k], keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c"))
                # wandb.log({f'final_example_dtw_distance_{ylab_list[k]}': alignment.distance})
                # dtw_normalizeddistance_sum += alignment.normalizedDistance
                # dtw_distance_sum += alignment.distance
                self.dtw_plot_for_thesis(axs_dtw[k], query[:, dof], reference[:, dof], matches, ylab=ylab_list[k], k=k, fig=fig_dtw, half=half)
                
                if k == 7:
                    axs_dtw[k].set_xlabel('Time Step (environment)')

            fig_dtw.set_size_inches(10, 16)
            fig_dtw.tight_layout()
            fig_dtw.subplots_adjust(top=0.97)#, bottom=0.07, left=0.06, right=0.93)

            path_data = os.path.join(
                path, 
                f'Pictures/{timestep}/{env_name}/plot_dtw_10x16_8lines_1.pdf'
            )
            if not os.path.exists(os.path.dirname(path_data)):
                os.makedirs(os.path.dirname(path_data))

            fig_dtw.savefig(path_data)
            plt.close(fig_dtw)

    def dtw_plot_for_thesis(self, ax, xts, yts, match_indices,
                  offset=2,
                  ts_type="l",                  
                  match_col="gray",
                  xlab="Index",
                  ylab="Query value",
                  k=None, fig=None, half=False,
                  **kwargs):
        
        from matplotlib import collections  as mc

        # if xts is None or yts is None:
        #     try:
        #         xts = d.query
        #         yts = d.reference
        #     except:
        #         raise ValueError("Original timeseries are required")
        

        offset = -offset

        xtimes = numpy.arange(len(xts))
        ytimes = numpy.arange(len(yts))
        
        # ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        
        ax.plot(xtimes, numpy.array(xts), color='k', label="policy", **kwargs)
        ax.plot(ytimes, numpy.array(yts) - offset, color='tab:blue', label="healthy", **kwargs)      # Plot with offset applied
        ax.grid()

        if offset != 0:
            # Create an offset axis
            ax2 = ax.twinx()
            ax2.tick_params('y', colors='tab:blue')
            ql, qh = ax.get_ylim()
            ax2.set_ylim(ql + offset, qh + offset)

        # https://stackoverflow.com/questions/21352580/matplotlib-plotting-numerous-disconnected-line-segments-with-different-colors
        # if match_indices is None:
        #     idx = numpy.linspace(0, len(d.index1) - 1)
        # elif not hasattr(match_indices, "__len__"):
        #     idx = numpy.linspace(0, len(d.index1) - 1, num=match_indices)
        # else:
        #     idx = match_indices
        # idx = numpy.array(idx).astype(int)


        col = []
        for i in match_indices:
            col.append([(i[1], xts[i[1]]),
                        (i[0], -offset + yts[i[0]])])
        lc = mc.LineCollection(col, linewidths=1, linestyles=":", colors=match_col)
        ax.add_collection(lc)

        if half: 
            if k == 0:
                fig.legend(loc="upper center", ncol=3)
            if k == 4:
                fig.legend(loc="upper center", ncol=3)
        else:
            if k == 0:
                fig.legend(loc="upper center", ncol=3)


    def plot_traj_for_thesis(self,half=False, env_name=None, path=None, timestep=None):
        
        titles = ['Pelvis Tilt (rad)', 
                  'Pelvis Height (m)', 
                  'Right Hip (rad)', 
                  'Right Knee (rad)', 
                  'Right Ankle (rad)', 
                  'Left Hip (rad)', 
                  'Left Knee (rad)', 
                  'Left Ankle (rad)',]
        thesis_grid = np.linspace(0,1000,1000, dtype=int)
        if half:
        
            fig, axs = plt.subplots(4,2)
            axs = np.array(axs)
            # fig.suptitle(f'Dofs: Healthy vs. Patho vs. Trained_{trained_steps}_steps')

            for idx, dof in enumerate([0, 2, 3, 4, 5, 6, 7, 8]):
                x = int(idx / 2)
                y = idx % 2
                ax = axs[x][y]
                normal = np.array(self.buffer_normal.states[thesis_grid, :][:, dof].cpu())
                patho = np.array(self.buffer_patho.states[thesis_grid, :][:, dof].cpu())
                trained = np.array(self.states_trained.cpu())[thesis_grid, :][:, dof]
                # expert_accom = np.array(self.states_expert.cpu())[self.grids, :][:, idx]
                ax.plot(thesis_grid, normal, label="healthy")
                ax.plot(thesis_grid, patho, label="pathological")
                ax.plot(thesis_grid, trained, label="policy")
                # ax.plot(self.grids, expert_accom, label="expert_accom")
                #ax.set_yticks(np.arange(normal.min(), normal.max(), 1))
                # ax.legend(loc='upper right')
                ax.grid()

                if idx == 0:
                    fig.legend(loc="upper center", ncol=3)
                if idx == 6 or idx == 7:
                    ax.set_xlabel('Time Step (environment)')

                if titles is not None:
                    ax.set_ylabel(titles[idx])
                
            fig.set_size_inches(10, 8)
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            plt.close(fig)

        else:
            '''
            fig_1, axs_1 = plt.subplots(8, 1)
            fig_2, axs_2 = plt.subplots(4, 1)
            for idx, dof in enumerate([0, 2, 3, 4, 5, 6, 7, 8]):
                if idx // 4 == 0:
                    axs = axs_1
                    fig = fig_1
                elif idx // 4 == 1:
                    axs = axs_2
                    fig = fig_2
                ax = axs[idx%4]
                normal = np.array(self.buffer_normal.states[thesis_grid, :][:, dof].cpu())
                patho = np.array(self.buffer_patho.states[thesis_grid, :][:, dof].cpu())
                trained = np.array(self.states_trained.cpu())[thesis_grid, :][:, dof]
                ax.plot(thesis_grid, normal, label="healthy")
                ax.plot(thesis_grid, patho, label="pathological")
                ax.plot(thesis_grid, trained, label="policy")
                ax.grid()

                if idx == 0 or idx == 4:
                    fig.legend(loc="upper center", ncol=3)
                if idx == 3 or idx == 7:
                    ax.set_xlabel('Time Step (environment)')

                if titles is not None:
                    ax.set_ylabel(titles[idx])
                
            fig_1.set_size_inches(10, 16)
            fig_2.set_size_inches(10, 8)
            fig_1.subplots_adjust(top=0.95, bottom=0.09, left=0.06, right=0.98)
            fig_2.subplots_adjust(top=0.95, bottom=0.09, left=0.06, right=0.98)
        
        fig_1.savefig(os.path.expanduser("~") + f'/Pictures/plot_10x16_8lines_1.pdf')
        fig_2.savefig(os.path.expanduser("~") + f'/Pictures/plot_10x16_8lines_2.pdf')
            '''
            fig, axs = plt.subplots(8, 1)
            for idx, dof in enumerate([0, 2, 3, 4, 5, 6, 7, 8]):
                ax = axs[idx]
                normal = np.array(self.buffer_normal.states[thesis_grid, :][:, dof].cpu())
                patho = np.array(self.buffer_patho.states[thesis_grid, :][:, dof].cpu())
                trained = np.array(self.states_trained.cpu())[thesis_grid, :][:, dof]
                ax.plot(thesis_grid, normal, label="healthy")
                ax.plot(thesis_grid, patho, label="pathological")
                ax.plot(thesis_grid, trained, label="policy")
                ax.grid()

                if idx == 0:
                    fig.legend(loc="upper center", ncol=3)
                if idx == 7:
                    ax.set_xlabel('Time Step (environment)')

                if titles is not None:
                    ax.set_ylabel(titles[idx])
                
            fig.set_size_inches(10, 16)
            fig.tight_layout()
            fig.subplots_adjust(top=0.97)#, bottom=0.09, left=0.06, right=0.98)
            path_data = os.path.join(
                path, 
                f'Pictures/{timestep}/{env_name}/plot_10x16_8lines_1.pdf'
            )
            if not os.path.exists(os.path.dirname(path_data)):
                os.makedirs(os.path.dirname(path_data))
        
            fig.savefig(path_data)
            plt.close(fig)

    ###########################################

    def plot_traj(self, titles=None, trained_steps=None, mean=None, var=None):

        for idx in range(self.buffer_normal.states[0].shape[0]):
            self.i += int(idx / 25) #fig_{i}
            if idx % 25 == 0:
                globals()[f'fig_{self.i}'], globals()[f'axs_{self.i}'] = plt.subplots(5,5)
                globals()[f'axs_{self.i}'] = np.array(globals()[f'axs_{self.i}'])

                globals()[f'fig_{self.i}'].suptitle(f'Dofs: Healthy vs. Patho vs. Trained_{trained_steps}_steps')

            x = int((idx % 25) / 5)
            y = (idx % 25) % 5
            ax = globals()[f'axs_{self.i}'][x][y]
            normal = np.array(self.buffer_normal.states[self.grids, :][:, idx].cpu())
            patho = np.array(self.buffer_patho.states[self.grids, :][:, idx].cpu())
            trained = np.array(self.states_trained.cpu())[self.grids, :][:, idx]
            # expert_accom = np.array(self.states_expert.cpu())[self.grids, :][:, idx]
            ax.plot(self.grids, normal, label="normal")
            ax.plot(self.grids, patho, label="patho")
            ax.plot(self.grids, trained, label="trained")
            # ax.plot(self.grids, expert_accom, label="expert_accom")
            #ax.set_yticks(np.arange(normal.min(), normal.max(), 1))
            # ax.legend(loc='upper right')
            ax.grid()

            if titles is not None:
                ax.set_title(titles[idx])
            
            plt.subplots_adjust(left=0.02, bottom=0.05, right=0.97, top=0.94, wspace=0.21, hspace=0.45)
            #globals()[f'fig_{i}'].set_size_inches(16, 10)

        # self.states_trained = torch.empty((1, self.buffer_normal.states.shape[1]))
        self.initial = True
        self.traj_cnt = self.i
        self.i += 1

        #for i in range(int((self.buffer_normal.states[0].shape[0] / 25) + 1)): 
        #    globals()[f'fig_{i}'].savefig(f'/home/zhang/SCONE/results/{path}/{trained_steps}_steps_eval_{i}.png', dpi=globals()[f'fig_{i}'].dpi)

        # plt.show()

    def plot_muscle_force(self, titles=None, trained_steps=None, path=None):

        for idx in range(self.buffer_normal.muscle_forces[0].shape[0]):
            self.i += int(idx / 25) #fig_{i}
            if idx % 25 == 0:
                globals()[f'fig_{self.i}'], globals()[f'axs_{self.i}'] = plt.subplots(5,5)
                globals()[f'axs_{self.i}'] = np.array(globals()[f'axs_{self.i}'])

                globals()[f'fig_{self.i}'].suptitle(f'Muscle forces: Healthy vs. Patho vs. Trained_{trained_steps}_steps')

            x = int((idx % 25) / 5)
            y = (idx % 25) % 5
            ax = globals()[f'axs_{self.i}'][x][y]
            normal = np.array(self.buffer_normal.muscle_forces[self.grids, :][:, idx].cpu())
            patho =np.array(self.buffer_patho.muscle_forces[self.grids, :][:, idx].cpu())
            trained = np.array(self.muscle_force_trained.cpu())[self.grids, :][:, idx]
            # expert_accom = np.array(self.muscle_force_expert.cpu())[self.grids, :][:, idx]
            ax.plot(self.grids, normal, label="normal")
            ax.plot(self.grids, patho, label="patho")
            ax.plot(self.grids, trained, label="trained")
            # ax.plot(self.grids, expert_accom, label="expert_accom")
            #ax.set_yticks(np.arange(normal.min(), normal.max(), 1))
            ax.legend(loc='upper right')

            if titles is not None:
                ax.set_title(titles[idx])
            
            plt.subplots_adjust(left=0.02, bottom=0.05, right=0.97, top=0.94, wspace=0.21, hspace=0.45)
            #globals()[f'fig_{i}'].set_size_inches(16, 10)

        # self.muscle_force_trained = torch.empty((1, self.buffer_normal.muscle_forces.shape[1]))
        self.initial = True
        self.muscle_force_cnt = self.i
        self.i += 1

        #for i in range(int((self.buffer_normal.states[0].shape[0] / 25) + 1)): 
        #    globals()[f'fig_{i}'].savefig(f'/home/zhang/SCONE/results/{path}/{trained_steps}_steps_eval_{i}.png', dpi=globals()[f'fig_{i}'].dpi)

        # plt.show()
    

    def plot_excitation(self, titles=None, trained_steps=None, path=None):

        for idx in range(self.buffer_normal.excitations[0].shape[0]):
            self.i += int(idx / 25) #fig_{i}
            if idx % 25 == 0:
                globals()[f'fig_{self.i}'], globals()[f'axs_{self.i}'] = plt.subplots(5,5)
                globals()[f'axs_{self.i}'] = np.array(globals()[f'axs_{self.i}'])

                globals()[f'fig_{self.i}'].suptitle(f'Excitations: Healthy vs. Patho vs. Trained_{trained_steps}_steps')

            x = int((idx % 25) / 5)
            y = (idx % 25) % 5
            ax = globals()[f'axs_{self.i}'][x][y]
            normal = np.array(self.buffer_normal.excitations[self.grids, :][:, idx].cpu())
            patho =np.array(self.buffer_patho.excitations[self.grids, :][:, idx].cpu())
            trained = np.array(self.excitation_trained.cpu())[self.grids, :][:, idx]
            # expert_accom = np.array(self.excitation_expert.cpu())[self.grids, :][:, idx]
            ax.plot(self.grids, normal, label="normal")
            ax.plot(self.grids, patho, label="patho")
            ax.plot(self.grids, trained, label="trained")
            # ax.plot(self.grids, expert_accom, label="expert_accom")
            #ax.set_yticks(np.arange(normal.min(), normal.max(), 1))
            ax.legend(loc='upper right')

            if titles is not None:
                ax.set_title(titles[idx])
            
            plt.subplots_adjust(left=0.02, bottom=0.05, right=0.97, top=0.94, wspace=0.21, hspace=0.45)
            #globals()[f'fig_{i}'].set_size_inches(16, 10)

        # self.excitation_trained = torch.empty((1, self.buffer_normal.excitations.shape[1]))
        self.initial = True
        self.excitation_cnt = self.i
        self.i += 1

        #for i in range(int((self.buffer_normal.states[0].shape[0] / 25) + 1)): 
        #    globals()[f'fig_{i}'].savefig(f'/home/zhang/SCONE/results/{path}/{trained_steps}_steps_eval_{i}.png', dpi=globals()[f'fig_{i}'].dpi)

        # plt.show()

    

    def plot_activation(self, titles=None, trained_steps=None, path=None):

        for idx in range(self.buffer_normal.activations[0].shape[0]):
            self.i += int(idx / 25) #fig_{i}
            if idx % 25 == 0:
                globals()[f'fig_{self.i}'], globals()[f'axs_{self.i}'] = plt.subplots(5,5)
                globals()[f'axs_{self.i}'] = np.array(globals()[f'axs_{self.i}'])

                globals()[f'fig_{self.i}'].suptitle(f'Activations: Healthy vs. Patho vs. Trained_{trained_steps}_steps')

            x = int((idx % 25) / 5)
            y = (idx % 25) % 5
            ax = globals()[f'axs_{self.i}'][x][y]
            normal = np.array(self.buffer_normal.activations[self.grids, :][:, idx].cpu())
            patho =np.array(self.buffer_patho.activations[self.grids, :][:, idx].cpu())
            trained = np.array(self.activation_trained.cpu())[self.grids, :][:, idx]
            # expert_accom = np.array(self.activation_expert.cpu())[self.grids, :][:, idx]
            ax.plot(self.grids, normal, label="normal")
            ax.plot(self.grids, patho, label="patho")
            ax.plot(self.grids, trained, label="trained")
            # ax.plot(self.grids, expert_accom, label="expert_accom")
            #ax.set_yticks(np.arange(normal.min(), normal.max(), 1))
            ax.legend(loc='upper right')

            if titles is not None:
                ax.set_title(titles[idx])
            
            plt.subplots_adjust(left=0.02, bottom=0.05, right=0.97, top=0.94, wspace=0.21, hspace=0.45)
            #globals()[f'fig_{i}'].set_size_inches(16, 10)

        # self.activation_trained = torch.empty((1, self.buffer_normal.activations.shape[1]))
        self.initial = True
        self.activation_cnt = self.i
        self.i += 1

        #for i in range(int((self.buffer_normal.states[0].shape[0] / 25) + 1)): 
        #    globals()[f'fig_{i}'].savefig(f'/home/zhang/SCONE/results/{path}/{trained_steps}_steps_eval_{i}.png', dpi=globals()[f'fig_{i}'].dpi)

        # plt.show()

    def upload(self, env_name):
        for j in range(self.traj_cnt + 1): 
            #img = mplfig_to_npimage(globals()[f'fig_{i}'])
            img = globals()[f'fig_{j}']
            globals()[f'fig_{j}'].set_size_inches(16, 10)
            wandb_img = wandb.Image(img)
            wandb.log({f'traj_analysis_{env_name}': wandb_img})
            plt.close(globals()[f'fig_{j}'])
        
        for j in range(self.traj_cnt + 1, self.muscle_force_cnt + 1): 
            #img = mplfig_to_npimage(globals()[f'fig_{i}'])
            img = globals()[f'fig_{j}']
            globals()[f'fig_{j}'].set_size_inches(16, 10)
            wandb_img = wandb.Image(img)
            wandb.log({f'muscle_force_analysis_{env_name}': wandb_img})
            plt.close(globals()[f'fig_{j}'])
        
        for j in range(self.muscle_force_cnt + 1, self.excitation_cnt + 1): 
            #img = mplfig_to_npimage(globals()[f'fig_{i}'])
            img = globals()[f'fig_{j}']
            globals()[f'fig_{j}'].set_size_inches(16, 10)
            wandb_img = wandb.Image(img)
            wandb.log({f'excitation_analysis_{env_name}': wandb_img})
            plt.close(globals()[f'fig_{j}'])
        
        for j in range(self.excitation_cnt + 1, self.activation_cnt + 1): 
            #img = mplfig_to_npimage(globals()[f'fig_{i}'])
            img = globals()[f'fig_{j}']
            globals()[f'fig_{j}'].set_size_inches(16, 10)
            wandb_img = wandb.Image(img)
            wandb.log({f'activation_analysis_{env_name}': wandb_img})
            plt.close(globals()[f'fig_{j}'])

        self.i = 0
        


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--buffer', type=str, required=False)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    args.buffer = "./buffers/sconewalk_origin_motored_normal_h0914-v3/size1000000_ori_freq1.0_new_freq2.0.pth"
    comp_states_nextstates(args)

