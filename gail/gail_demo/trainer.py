import os
import torch
import torch.nn.functional as F
import numpy as np
import time
# from time import time, sleep
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pdb
import wandb
import matplotlib.pyplot as plt
import gym
import random 
from datetime import date
from drawing import Carrier, Dof_store
from gait_cycle import determine_gait_cycle, transform_time_to_gait_cycle
from frequency import plot_simulation_in_time
from rewards import Dissimilarity
from .algo.gail import Chunking

available_init_states = {
    'sconewalk_origin_motored_shorthamstring_h0914-v3': [554, 667, 1006, 1119, 1121, 1258, 1570, 1685, 2021, 2134, 2149, 2585, 2700, 2724, 3264, 3600, 4166, 4615], 
    'sconewalk_origin_motored_plantarweakness_h0914-v3': [0, 3, 4, 5, 6, 40, 41, 42, 47, 105, 238, 239], 
    'sconewalk_origin_motored_hamstringweakness_h0914-v3': [3, 4, 5, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], 
    'sconewalk_origin_motored_hamstringweakness_new_h0914-v3': [3, 9, 10, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23], # 11, 12,  28, 29, 31, 32, 33
    'sconewalk_origin_motored_iliopsoasweakness_h0914-v3': [0, 21, 37, 940, 1818, 2833, 2858, 3114, 3848, 3985]
}

def normalize2d(batch, mean, var):
    norm_batch = batch.copy()
    for i in range(batch.shape[1]):
        batch[:, i] = (batch[:, i] - mean[i]) / (var[i] + 1e-6)
    return norm_batch

class Trainer:

    def __init__(self, env, env_test, algo, log_dir, seed=0, num_steps=10**5,
                 eval_interval=10**3, num_eval_episodes=1, draw=False, agent_env_id=None, expert_env_id=None, max_step=10**6, device=None, animating=False, 
                 header_num=3, obs_history=4, act_chunked=4,
                 ):
        super().__init__()

        # Env to collect samples.
        self.env = env
        # self.env.seed(seed)

        # Env for evaluation.
        self.env_test = env_test
        # self.env_test.seed(2**31-seed)
        self.seed = seed

        self.algo = algo
        self.log_dir = log_dir

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

        self.draw = draw
        self.animating = animating
        self.agent_env_id = agent_env_id
        self.expert_env_id = expert_env_id
        self.device = device

        self.max_step = max_step
        self.loss_MSE = torch.nn.MSELoss().to("cuda" if torch.cuda.is_available else "cpu")
        self.loss_MAE = torch.nn.L1Loss().to("cuda" if torch.cuda.is_available else "cpu")

        self.env_list = ['sconewalk_origin_motored_hamstringweakness_new_h0914-v3', 
                         'sconewalk_origin_motored_iliopsoasweakness_h0914-v3', 
                         'sconewalk_origin_motored_shorthamstring_h0914-v3']
        
        self.env_id_dict = {'sconewalk_origin_motored_hamstringweakness_new_h0914-v3': torch.from_numpy(np.array([0])), 
                           'sconewalk_origin_motored_iliopsoasweakness_h0914-v3': torch.from_numpy(np.array([1])), 
                           'sconewalk_origin_motored_shorthamstring_h0914-v3': torch.from_numpy(np.array([2])), }
        
        self.weights_cnt = [0, 0, 0]

        self.env_abbr = {'sconewalk_origin_motored_hamstringweakness_new_h0914-v3': 'hamwl', 
                         'sconewalk_origin_motored_iliopsoasweakness_h0914-v3': 'iliowl', 
                         'sconewalk_origin_motored_shorthamstring_h0914-v3': 'shortham', }

        self.header_num = header_num
        self.obs_history = obs_history
        self.act_chunked = act_chunked
        

    def train(self,config=None):
        
        self.config = {
            "epoch": self.num_steps, 
            "eval_interval": self.eval_interval, 
            "observations": "dof",
        }

        self.run = wandb.init(
            project="MA_rubin", 
            # name="hamstringweak_norm_shorttime_newmodel_sband1.9_dtw2e3_04101433", 
           # config= self.config
        )
        # self.init_states = np.array([-0.1331,  0.0000,  0.8925,  0.6445, -0.4584, -0.0520, -0.2192, -0.2109,
        #                             0.1575,  0.1266,  1.0792,  0.0243,  0.6651, -3.4124,  1.9854,  0.9138,
        #                             -4.0062, -1.1726])
        self.init_states = np.array([-0.0543,  0.0000,  0.9277,  0.3545, -0.1350,  0.0291,  0.0416, -1.1989,
                                    -0.1598, -0.5721,  1.0000,  0.0546, -0.5764,  0.1759,  0.9889,  4.9067,
                                    -3.5979,  0.6333])

        # Time to start training.
        self.start_time = time.time()
        self.start_time_fixed = f'{date.today()}_{time.strftime("%H_%M")}'
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        # distri = [z - min(self.weights_cnt) + 1 for z in self.weights_cnt]
        # env_name = random.choices(self.env_list, weights=distri, k=1)
        # env_name = env_name[0]
        # self.weights_cnt[self.env_id_dict[env_name]] -= 1
        # env_name = self.env_list[1]
        env_cnt = 0
        env_name = self.env_list[env_cnt % 3]
        cur_env = self.env[env_name]
        state, state_ori = cur_env.reset(norm=True)
        # Add one hot
        # state = np.concat((F.one_hot(self.env_id_dict[env_name], num_classes=3).squeeze().numpy(), state))
        # Add embedding
        # state = torch.concat((self.embed(self.env_id_dict[env_name]).squeeze(), torch.from_numpy(state)))
        step = 0
        env_cnt += 1

        for _ in tqdm(range(1, self.num_steps + 1)):
            step += 1
            # Pass to the algorithm to update state and episode timestep.
            state, t, _, done = self.algo.step(env_name, cur_env, state, t, step, self.init_states, norm=True)
            # print("Env name: ", env_name)
            
            if done: 
                # print("Switch at step ", step)
                # distri = [z - min(self.weights_cnt) + 1 for z in self.weights_cnt]
                # env_name = random.choices(self.env_list, weights=distri, k=1)
                # env_name = env_name[0]
                # self.weights_cnt[self.env_id_dict[env_name]] -= 1
                # cur_env = self.env[env_name]
                state, state_ori = cur_env.reset(norm=True)
                # print("name: ", env_name)
            # if self.algo.buffer_sum[env_name]._p % self.algo.rollout_length == 0 and not self.algo.buffer_sum[env_name].clean:
                

            # Update the algorithm whenever ready.
            name, update = self.algo.is_update(env_name)
            if update: 
                update_env = self.env[name]
                self.algo.update(self.writer, update_env, name, config)
                # print(name)
                # for k in self.env_list:
                #     print(self.algo.embed(self.env_id_dict[k]).squeeze())
                env_name = self.env_list[env_cnt % 3]
                cur_env = self.env[env_name]
                state, state_ori = cur_env.reset(norm=True)
                env_cnt += 1
                # pdb.set_trace()

            # Evaluate regularly.
            if step % self.eval_interval == 0:
                # self.evaluate(step,init_range=400)
                for env_name in self.env_list:
                # env_name = self.env_list[0]
                    eval_dissimilarity, eval_objective = self.evaluate(env_name, step, init_range=4000, embedding=self.algo.embed, thesis=True, pandas_thesis=True, path=os.path.expanduser("~")+'/Thesis/minibatch_obs1_act4_minibatch1e4_epoch50_randombatch')
                    # wandb.log({f'eval_dissimilarity_{env_name}': eval_dissimilarity}) self.algo.embed
                    # wandb.log({f'eval_objective_{env_name}': eval_objective})
                # distri = [z - min(self.weights_cnt) + 1 for z in self.weights_cnt]
                # env_name = random.choices(self.env_list, weights=distri, k=1)
                # self.weights_cnt[self.env_id_dict[env_name]] -= 1
                # cur_env = self.env[env_name]
                # state, state_ori = cur_env.reset(norm=True)
                # # Add embedding / one hot
                # state = np.concat((F.one_hot(self.env_id_dict[env_name], num_classes=3).squeeze().numpy(), state))
                # self.algo.save_models(
                #     os.path.join(self.model_dir, f'step{step}'))

        # Wait for the logging to be finished.
        #sleep(10)
        # dissimilarity, objective = self.calc_dissimilarity(step, init_range=10**4)
        # return dissimilarity, objective

        self.run.finish()
    
    # time = []
    # trajectories = []
    # grf = []
    # actions = []

    # # Frequency analysis initialization
    # time.append(self.env_test.time)
    # trajectories.append([self.env_test.model.dofs()[i].pos() for i in range(len(self.env_test.model.dofs()))])
    # grf.append(
    #     [
    #         self.env_test.model.legs()[0].contact_force().array(),
    #         self.env_test.model.legs()[1].contact_force().array(),
    #     ]
    # )
    # actions.append(np.zeros(*self.env.action_space.shape))

    
    def evaluate(self, env_name, step, init_range, embedding, thesis=False, pandas_thesis=False, path=None):
        dissimilarity_list = []
        objective_list = []
        mse_list = []
        mae_list = []
        dtw_list = []
        i_list = []
        expert_analysis_val = False
        epi_dissimilarity = 0.0
        sum_dissimilarity = 0.0
        epi_objective = 0.0
        sum_objective = 0.0
        sum_mse_pos = 0.0
        sum_mae_pos = 0.0
        sum_mse_vel = 0.0
        sum_mae_vel = 0.0
        mean_dissimilarity = 0.0
        mean_objective = 0.0
        sum_dtw_normalizeddistance_pos = 0.0
        sum_dtw_normalizeddistance_vel = 0.0
        cnt = -1000
        rollout_cnt = 0

        # env_name = random.choice(self.env_list)
        # env_name = 'sconewalk_origin_motored_hamstringweakness_new_h0914-v3'

        carrier_eval = Carrier(env_name, self.expert_env_id, expert_analysis_val, max_step=self.max_step)
        dof_storer = Dof_store()
        init_states = random.sample(available_init_states[env_name], 10)

        self.env_test[env_name].obs_mean = self.env[env_name].obs_mean
        self.env_test[env_name].obs_var = self.env[env_name].obs_var
        self.env_test[env_name].obs_cnt = self.env[env_name].obs_cnt

        for i in init_states:
            carrier_eval.clear_agent()
            init_state = carrier_eval.buffer_normal.states[i].cpu().numpy()
            self.env_test[env_name].store_next_episode()
            state_agent, state_agent_ori, info_agent = self.env_test[env_name].reset(return_info=True, init_values=init_state, norm=True)
            action_agent = np.zeros(4)
            reward = 0.0
            # carrier_eval.append_agent(state_agent, action_agent, reward, info_agent['muscle_force'], info_agent['excitation'], info_agent['activation'], 
            #                           info_agent['rew_dict']['grf'], info_agent['rew_dict']['gaussian_vel'], info_agent['meta_cost'])
            
            chunker = Chunking(self.header_num + state_agent.shape[0], action_agent.shape[0], self.obs_history, self.act_chunked, self.device)
            episode_return = 0.0
            done_agent = False
            done_expert = False
            ep = 0
            cnt += 1

            while (not done_agent):
                ep += 1
                # Add one hot
                # state_agent = np.concat((F.one_hot(self.env_id_dict[env_name], num_classes=3).squeeze().numpy(), state_agent))
                # Add embdedding
                if embedding is not None:
                    state_agent = np.concatenate((embedding(self.env_id_dict[env_name]).squeeze().detach().numpy(), state_agent))
                
                if ep == 1:
                    chunker.obs_trunker_init(torch.from_numpy(state_agent))
                else: chunker.obs_trunker_update(torch.from_numpy(state_agent))
                state_agent = chunker.get_obs()
                action_agent = self.algo.exploit(state_agent)
                if ep == 1:
                    chunker.act_trunker_init(action_agent)
                else: chunker.act_trunker_update(action_agent)
                action_agent = chunker.get_act()
                carrier_eval.append_agent(state_agent_ori, action_agent, reward, info_agent['muscle_force'], info_agent['excitation'], info_agent['activation'], 
                                          info_agent['rew_dict']['grf'], info_agent['rew_dict']['gaussian_vel'], info_agent['meta_cost'])
                state_agent, state_agent_ori, reward, done_agent, info_agent = self.env_test[env_name].step(action_agent, norm=True)
                
                episode_return += reward


                # print("Env_name ", env_name)
                # print("1. Fall ", self.env_test[env_name].model.com_pos().y)
                # print("2. Fall ", self.env_test[env_name].model.bodies()[self.env_test[env_name].head_body_idx].com_pos().y)
                # pdb.set_trace()
                
                
                if ep >= self.max_step - 2: break
            

            epi_dissimilarity, epi_objective, epi_mse_pos, epi_mae_pos, epi_mse_vel, epi_mae_vel, epi_dtw_normalizeddistance_pos, epi_dtw_normalizeddistance_vel = carrier_eval.calc_dissimilar(i, init_range)
            # dissimilarity_list.append(epi_dissimilarity.item())
            # objective_list.append(epi_objective.item())
            mse_list.append(epi_mse_pos.item())
            mae_list.append(epi_mae_pos.item())
            dtw_list.append(epi_dtw_normalizeddistance_pos)
            i_list.append(i)
            # pdb.set_trace()
            
            

            # carrier_eval.save(os.path.join(
            #     'amam_dissi_3', 
            #     f'corrected_agent_{step}_{cnt}.pth'
            # ))

            if pandas_thesis:
                # assert(path != None)
                # path_data = os.path.join(
                #     path, 
                #     env_name, 
                #     'loss',
                #     f'loss_{self.seed}_{rollout_cnt}_{step}.pth'
                # )
                # if not os.path.exists(os.path.dirname(path_data)):
                #     os.makedirs(os.path.dirname(path_data))

                # torch.save({
                #     'dtw': torch.Tensor(dtw_list), 
                #     'mse': torch.Tensor(mse_list), 
                #     'mae': torch.Tensor(mae_list), 
                #     'i': torch.Tensor(i_list)
                # }, path_data)

                carrier_eval.save(os.path.join(
                    path, 
                    env_name, 
                    'observations', 
                    f'corrected_agent_{self.seed}_{rollout_cnt}_{step}.pth'
                ))

            rollout_cnt += 1
            sum_dissimilarity += epi_dissimilarity
            sum_objective += epi_objective
            sum_mse_pos += epi_mse_pos
            sum_mae_pos += epi_mae_pos
            sum_mse_vel += epi_mse_vel
            sum_mae_vel += epi_mae_vel
            sum_dtw_normalizeddistance_pos += epi_dtw_normalizeddistance_pos
            sum_dtw_normalizeddistance_vel += epi_dtw_normalizeddistance_vel
            # self.env_test.gail_write_now(cnt)
        
        # mean_dissimilarity = sum_dissimilarity / 10
        # mean_objective = sum_objective / 10
        # wandb.log({'mean_dissimilarity': mean_dissimilarity})
        # wandb.log({'mean_objective': mean_objective})
        mean_mse_pos = sum_mse_pos / 10
        mean_mae_pos = sum_mae_pos / 10
        mean_dtw_normalizeddistance_pos = sum_dtw_normalizeddistance_pos / 10
        mean_mse_vel = sum_mse_vel / 10
        mean_mae_vel = sum_mae_vel / 10
        mean_dtw_normalizeddistance_vel = sum_dtw_normalizeddistance_vel / 10
        wandb.log({f'mse_pos_{env_name}': mean_mse_pos})
        wandb.log({f'mae_pos_{env_name}': mean_mae_pos})
        wandb.log({f'dtw_normalizeddistance_pos_{env_name}': mean_dtw_normalizeddistance_pos})

        # wandb.log({f'mse_vel_{env_name}': mean_mse_vel})
        # wandb.log({f'mae_vel_{env_name}': mean_mae_vel})
        # wandb.log({f'dtw_normalizeddistance_vel_{env_name}': mean_dtw_normalizeddistance_vel})

        # pdb.set_trace()
        # Save data
        # path = os.path.join(
        #     'amam_dissi_5', 
        #     f'loss_{step}_{cnt}.pth'
        # )

        # if not os.path.exists(os.path.dirname(path)):
        #     os.makedirs(os.path.dirname(path))

        # torch.save({
        #     'dissimilarity': torch.Tensor(dissimilarity_list), 
        #     'objective': torch.Tensor(objective_list), 
        #     'mse': torch.Tensor(mse_list), 
        #     'mae': torch.Tensor(mae_list), 
        #     'i': torch.Tensor(i_list)
        # }, path)
        # print("dissimilarity_list: ", dissimilarity_list)
        # print("mse_list: ", mse_list)
        # print("mae_list: ", mae_list)
        # print("i_list: ", i_list)

        if thesis:
            assert(path != None)
            path_data = os.path.join(
                path, 
                env_name, 
                'loss', 
                f'loss_{step}_{cnt}.pth'
            )
            if not os.path.exists(os.path.dirname(path_data)):
                os.makedirs(os.path.dirname(path_data))

            torch.save({
                'dtw': torch.Tensor(dtw_list), 
                'mse': torch.Tensor(mse_list), 
                'mae': torch.Tensor(mae_list), 
                'i': torch.Tensor(i_list)
            }, path_data)


        # Run the simulation again from beginning, log a video, with leg switched (left leg)
        carrier_eval.clear_agent()
        self.env_test[env_name].store_next_episode()
        state_agent, state_agent_ori, info_agent = self.env_test[env_name].reset(return_info=True, norm=True, brute_switch=True)
        action_agent = np.zeros(4)
        reward = 0.0
        # carrier_eval.append_agent(state_agent, action_agent, reward, info_agent['muscle_force'], info_agent['excitation'], info_agent['activation'], 
        #                               info_agent['rew_dict']['grf'], info_agent['rew_dict']['gaussian_vel'], info_agent['meta_cost'])
        episode_return = 0.0
        done_agent = False
        ep = 0
        chunker = Chunking(self.header_num + state_agent.shape[0], action_agent.shape[0], self.obs_history, self.act_chunked, self.device)

        while (not done_agent):
            ep += 1
            # Add one hot
            # state_agent = np.concat((F.one_hot(self.env_id_dict[env_name], num_classes=3).squeeze().numpy(), state_agent))
            # Add embdedding
            if embedding is not None:
                state_agent = np.concatenate((embedding(self.env_id_dict[env_name]).squeeze().detach().numpy(), state_agent))
            if ep == 1:
                chunker.obs_trunker_init(torch.from_numpy(state_agent))
            else: chunker.obs_trunker_update(torch.from_numpy(state_agent))
            state_agent = chunker.get_obs()
            action_agent = self.algo.exploit(state_agent)
            if ep == 1:
                chunker.act_trunker_init(action_agent)
            else: chunker.act_trunker_update(action_agent)
            action_agent = chunker.get_act()
            carrier_eval.append_agent(state_agent_ori, action_agent, reward, info_agent['muscle_force'], info_agent['excitation'], info_agent['activation'], 
                                          info_agent['rew_dict']['grf'], info_agent['rew_dict']['gaussian_vel'], info_agent['meta_cost'])
            state_agent, state_agent_ori, reward, done_agent, info_agent = self.env_test[env_name].step(action_agent, norm=True)
            episode_return += reward
            if ep >= self.max_step - 2: break
        
        self.env_test[env_name].gail_write_now(step-1, self.env_abbr[env_name])


        # Run the simulation again from beginning, log a video, and also draw the plot. from original (right leg)
        carrier_eval.clear_agent()
        self.env_test[env_name].store_next_episode()
        state_agent, state_agent_ori, info_agent = self.env_test[env_name].reset(return_info=True, init_values=self.init_states, norm=True)
        action_agent = np.zeros(4)
        reward = 0.0
        # carrier_eval.append_agent(state_agent, action_agent, reward, info_agent['muscle_force'], info_agent['excitation'], info_agent['activation'], 
        #                               info_agent['rew_dict']['grf'], info_agent['rew_dict']['gaussian_vel'], info_agent['meta_cost'])
        episode_return = 0.0
        done_agent = False
        ep = 0
        chunker = Chunking(self.header_num + state_agent.shape[0], action_agent.shape[0], self.obs_history, self.act_chunked, self.device)

        while (not done_agent):
            ep += 1
            # Add embedding / one hot
            # state_agent = np.concat((F.one_hot(self.env_id_dict[env_name], num_classes=3).squeeze().numpy(), state_agent))
            # Add embdedding
            if embedding is not None:
                state_agent = np.concatenate((embedding(self.env_id_dict[env_name]).squeeze().detach().numpy(), state_agent))
            if ep == 1:
                chunker.obs_trunker_init(torch.from_numpy(state_agent))
            else: chunker.obs_trunker_update(torch.from_numpy(state_agent))
            state_agent = chunker.get_obs()
            action_agent = self.algo.exploit(state_agent)
            if ep == 1:
                chunker.act_trunker_init(action_agent)
            else: chunker.act_trunker_update(action_agent)
            action_agent = chunker.get_act()
            carrier_eval.append_agent(state_agent_ori, action_agent, reward, info_agent['muscle_force'], info_agent['excitation'], info_agent['activation'], 
                                          info_agent['rew_dict']['grf'], info_agent['rew_dict']['gaussian_vel'], info_agent['meta_cost'])
            # pdb.set_trace()
            dof_storer.append(state_agent_ori, info_agent['obs_tx'])
            state_agent, state_agent_ori, reward, done_agent, info_agent = self.env_test[env_name].step(action_agent, norm=True)
            episode_return += reward
            if ep >= self.max_step - 2: break
        
        self.env_test[env_name].gail_write_now(step, self.env_abbr[env_name])

        if self.animating:
            dof_storer.save(os.path.join(
                'animations',
                f'{self.agent_env_id}',
                f"{self.start_time_fixed}_{step}.pth",
            ))
        # To print out and save   
        if thesis: 
            carrier_eval.plot_traj_for_thesis(half=False, env_name=env_name, path=path, timestep=step)
            dtw_normalizeddistance_pos, _, _, _, matches, query, reference = carrier_eval.align_dtw(0, init_range, use_open_end=True)
            carrier_eval.save_plot_dtw_for_thesis(env_name, step, self.env_test[env_name].obs_name_list, reference, query, matches, half=False, path=path, timestep=step)
        # pdb.set_trace()
        # To print out and save  

        if self.draw:

            carrier_eval.calc_dissimilar(0, init_range) # To make sure ex, pi are correctly selected
            # Setteled: using expert as query and agent as reference
            dtw_normalizeddistance_pos, _, _, _, matches, query, reference = carrier_eval.align_dtw(0, init_range, use_open_end=True)
            carrier_eval.upload_plot_dtw(env_name, step, self.env_test[env_name].obs_name_list, reference, query, matches)
            # dtw_normalizeddistance_vel, _, _, _, matches, query, reference = carrier_eval.align_dtw(0, init_range,use_vel=True)
            # carrier_eval.upload_plot_dtw(env_name, step, self.env_test[env_name].obs_name_list, reference, query, matches, use_vel=True)
            carrier_eval.calc_plot_action(env_name, step)

            # This is for a comparison, run dtw indivisually to see if it works
            # carrier_eval.upload_plot_dtw_indivi(env_name, step, self.env_test[env_name].obs_name_list, 0, init_range)
        
        # carrier_eval.save(os.path.join(
        #     'amam_5', 
        #     f'corrected_agent_{step}.pth'
        # ))

        # final_dissimilarity, final_objective = carrier_eval.calc_dissimilar(i, init_range)
        # wandb.log({'final_dissimilarity': final_dissimilarity})
        # wandb.log({'final_objective': final_objective})

        # loss_interval = init_range
        # if init_range > carrier_eval._p:
        #     loss_interval = carrier_eval._p

        # agent_MSE = self.loss_MSE(carrier_eval.states_trained[:loss_interval], carrier_eval.buffer_normal.states[:loss_interval]) / loss_interval
        # patho_MSE = self.loss_MSE(carrier_eval.buffer_patho.states[:loss_interval], carrier_eval.buffer_normal.states[:loss_interval]) / loss_interval
        # agent_MAE = self.loss_MAE(carrier_eval.states_trained[:loss_interval], carrier_eval.buffer_normal.states[:loss_interval]) / loss_interval
        # patho_MAE = self.loss_MAE(carrier_eval.buffer_patho.states[:loss_interval], carrier_eval.buffer_normal.states[:loss_interval]) / loss_interval
        # # pdb.set_trace()
        # wandb.log({'agent_MSE': agent_MSE})
        # wandb.log({'patho_MSE': patho_MSE})
        # wandb.log({'agent_MAE': agent_MAE})
        # wandb.log({'patho_MAE': patho_MAE})
        

            if ep < 500:
                    carrier_eval.grids = np.linspace(0, ep - 2, 100, dtype=int)
            

            carrier_eval.plot_traj(self.env_test[env_name].obs_name_list, step, self.env_test[env_name].obs_mean, self.env_test[env_name].obs_var)
            carrier_eval.plot_muscle_force(self.env_test[env_name].muscle_name_list, step, self.env_test[env_name].output_dir)
            carrier_eval.plot_excitation(self.env_test[env_name].muscle_name_list, step, self.env_test[env_name].output_dir)
            carrier_eval.plot_activation(self.env_test[env_name].muscle_name_list, step, self.env_test[env_name].output_dir)
            carrier_eval.upload(env_name)
        print(f'Num steps: {step:<6}   '
            #   f'Dissimilarity: {dissimilarity:<5.1f}   '
              f'Return: {mean_mse_pos:<5.1f}   '
              f'Return: {mean_dtw_normalizeddistance_pos:<5.1f}   '
              f'Time: {self.time}')
        # log done
        
        return mean_dissimilarity, mean_objective
    

    def calc_dissimilarity(self, step, init_range):
        expert_analysis_val = False
        epi_dissimilarity = 0.0
        sum_dissimilarity = 0.0
        epi_objective = 0.0
        sum_objective = 0.0
        mean_dissimilarity = 0.0
        mean_objective = 0.0
        cnt = -1000
        carrier_dissimi = Carrier(self.agent_env_id, self.expert_env_id, expert_analysis_val, max_step=self.max_step)
        init_states = random.sample(available_init_states[self.agent_env_id], 10)

        for i in init_states:
            carrier_dissimi.clear_agent()
            init_state = carrier_dissimi.buffer_normal.states[i].cpu().numpy()
            self.env_test.store_next_episode()
            state_agent, info_agent = self.env_test.reset(return_info=True, init_values=init_state, norm=True, norm_reset=True)
            action_agent = np.zeros(4)
            reward = 0.0
            
            
            episode_return = 0.0
            done_agent = False
            done_expert = False
            ep = 0
            cnt += 1

            while (not done_agent):
                ep += 1
                action_agent = self.algo.exploit(state_agent)
                carrier_dissimi.append_agent(state_agent, action_agent, reward, info_agent['muscle_force'], info_agent['excitation'], info_agent['activation'], 
                                      info_agent['rew_dict']['grf'], info_agent['rew_dict']['gaussian_vel'], info_agent['meta_cost'])
                state_agent, reward, done_agent, info_agent = self.env_test.step(action_agent, norm=True)
                
                episode_return += reward
                
                # carrier_dissimi.append_agent(state_agent, info_agent['muscle_force'], info_agent['excitation'], info_agent['activation'], 
                #                           info_agent['rew_dict']['grf'], info_agent['rew_dict']['gaussian_vel'], info_agent['meta_cost'])
                
                if ep >= self.max_step - 2: break
               

            epi_dissimilarity, epi_objective, epi_mse, epi_mae = carrier_dissimi.calc_dissimilar(i, init_range)
           

            sum_dissimilarity += epi_dissimilarity
            sum_objective += epi_objective
            self.env_test.gail_write_now(cnt)
        
        mean_dissimilarity = sum_dissimilarity / 10
        mean_objective = sum_objective / 10
        
        return mean_dissimilarity, mean_objective




    @property
    def time(self):
        return str(timedelta(seconds=int(time.time() - self.start_time)))
