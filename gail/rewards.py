import numpy as np
import pdb



class Dissimilarity():
    def __init__(self):
        self.pi_rew_grf, self.ex_rew_grf = [], []
        self.pi_dof_pos, self.ex_dof_pos = [], []
        self.pi_dof_vel, self.ex_dof_vel = [], []
        self.pi_trunk_vel, self.ex_trunk_vel = [], []
        self.pi_meta_cost, self.ex_meta_cost = [], []

    def clear(self): 
        self.pi_rew_grf, self.ex_rew_grf = [], []
        self.pi_dof_pos, self.ex_dof_pos = [], []
        self.pi_dof_vel, self.ex_dof_vel = [], []
        self.pi_trunk_vel, self.ex_trunk_vel = [], []
        self.pi_meta_cost, self.ex_meta_cost = [], []
    
    def normalize2d(self, batch, mean, std):
        norm_batch = batch.copy()
        for i in range(batch.shape[1]):
            norm_batch[:, i] = (batch[:, i] - mean[i]) / (std[i] + 1e-6)
        return norm_batch
    
    def append(self, pi_grf, ex_grf, pi_dof_pos, pi_dof_vel, ex_dof_pos, ex_dof_vel, pi_trunk_vel, ex_trunk_vel, pi_meta_cost, ex_meta_cost): 
        self.pi_rew_grf.append(pi_grf)
        self.ex_rew_grf.append(ex_grf)
        self.pi_dof_pos.append(pi_dof_pos)
        self.pi_dof_vel.append(pi_dof_vel)
        self.ex_dof_pos.append(ex_dof_pos)
        self.ex_dof_vel.append(ex_dof_vel)
        self.pi_trunk_vel.append(pi_trunk_vel)
        self.ex_trunk_vel.append(ex_trunk_vel)
        self.pi_meta_cost.append(pi_meta_cost)
        self.ex_meta_cost.append(ex_meta_cost)
    
    def calc_dissimilar(self):
        self.ex_rew_grf = np.array(self.ex_rew_grf)
        self.ex_dof_pos = np.array(self.ex_dof_pos)
        self.ex_dof_vel = np.array(self.ex_dof_vel)
        self.pi_dof_pos = np.array(self.pi_dof_pos)
        self.pi_dof_vel = np.array(self.pi_dof_vel)
        self.ex_rew_grf_mean = self.ex_rew_grf.mean()
        self.ex_rew_grf_std = self.ex_rew_grf.std()
        # self.norm_ex_rew_grf = (self.ex_rew_grf - self.ex_rew_grf_mean) / self.ex_rew_grf_std
        # self.norm_pi_rew_grf = (self.pi_rew_grf - self.ex_rew_grf_mean) / self.ex_rew_grf_std

        self.ex_dof_pos_mean = self.ex_dof_pos.mean(axis=0)
        self.ex_dof_pos_std = self.ex_dof_pos.std(axis=0)
        self.ex_dof_vel_mean = self.ex_dof_vel.mean(axis=0)
        self.ex_dof_vel_std = self.ex_dof_vel.std(axis=0)
        # self.norm_ex_dof_pos = self.normalize2d(self.ex_dof_pos, self.ex_dof_pos_mean, self.ex_dof_pos_std)
        # self.norm_ex_dof_vel = self.normalize2d(self.ex_dof_vel, self.ex_dof_vel_mean, self.ex_dof_vel_std)
        # self.norm_pi_dof_pos = self.normalize2d(self.pi_dof_pos, self.ex_dof_pos_mean, self.ex_dof_pos_std)
        # self.norm_pi_dof_vel = self.normalize2d(self.pi_dof_vel, self.ex_dof_vel_mean, self.ex_dof_vel_std)

        # norm_dissimilarity = (((self.norm_pi_rew_grf - self.norm_ex_rew_grf)**2).sum() + ((self.norm_pi_dof_pos - self.norm_ex_dof_pos)**2).sum() + 
        #                  ((self.norm_pi_dof_vel - self.norm_ex_dof_vel)**2).sum() + (np.array(self.pi_trunk_vel).mean() - np.array(self.ex_trunk_vel).mean()) +
        #                  (np.array(self.pi_meta_cost).mean() - np.array(self.ex_meta_cost).mean()) )
        
        dissimilarity = ((((self.pi_rew_grf - self.ex_rew_grf))**2).sum() + ((self.pi_dof_pos - self.ex_dof_pos)**2).sum() + 
                         ((self.pi_dof_vel - self.ex_dof_vel)**2).sum() + 100 * (np.array(self.pi_trunk_vel).mean() - np.array(self.ex_trunk_vel).mean()) +
                         (np.array(self.pi_meta_cost).mean() - np.array(self.ex_meta_cost).mean()) )
        
        return dissimilarity