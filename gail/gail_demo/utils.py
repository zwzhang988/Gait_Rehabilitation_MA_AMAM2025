from tqdm import tqdm
import numpy as np
from scipy import interpolate
import torch
import pdb

from .buffer import Buffer, SerializedBuffer, QuerBuffer
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


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False


def add_random_noise(action, std):
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def trans_quer(buffer_path):

    buffer_ori = SerializedBuffer(
        path=buffer_path,
        device=torch.device("cpu")
    )
    pdb.set_trace()

    buffer = QuerBuffer(
        buffer_size=buffer_ori.states.size(0),
        state_shape=[buffer_ori.states.size(1)],
        action_shape=[buffer_ori.actions.size(1)],
        muscle_shape=buffer_ori.muscle_forces.size(1), 
        device=torch.device("cpu")
    )

    buffer.copy(buffer_ori.states, buffer_ori.actions, buffer_ori.rewards, buffer_ori.dones, buffer_ori.next_states, buffer_ori.muscle_forces, buffer_ori.excitations, 
                buffer_ori.activations, buffer_ori.rew_grfs, buffer_ori.trunk_vels, buffer_ori.meta_costs)
    
    for i in tqdm(range(buffer.states.shape[0])):
        buffer.quer_states[i] = deg2quat(buffer.states[i])
        buffer.quer_next_states[i] = deg2quat(buffer.next_states[i])

    return buffer

def collect_demo(env, algo, buffer_size, device, std, p_rand, seed=0, switch=False, switch_interval=3000):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        muscle_shape=len(env.model.muscles()), 
        device=device
    )

    total_return = 0.0
    num_episodes = 0
    switch_cnt = 0

    state, state_ori, info = env.reset(return_info=True)
    muscle_force = info['muscle_force']
    excitation = info['excitation']
    activation = info['activation']
    rew_grf = info['rew_dict']['grf']
    trunk_vel = info['rew_dict']['gaussian_vel']
    meta_cost = info['meta_cost']
    obs_tx = info['obs_tx']
    t = 0
    t_total = 0
    episode_return = 0.0

    for _ in tqdm(range(1, buffer_size + 1)):
        t += 1
        t_total += 1

        if np.random.rand() < p_rand:
            #action = env.action_space.sample()
            action = np.zeros(env.action_space.shape)
        else:
            action = algo.exploit(state)
            action = add_random_noise(action, std)

        next_state, next_state_ori, reward, done, info = env.step(action)
        mask = False if t == env._max_episode_steps else done
        buffer.append(state_ori, action, reward, mask, next_state_ori, muscle_force, excitation, activation, rew_grf, trunk_vel, meta_cost, obs_tx)
        episode_return += reward

        state, state_ori, muscle_force, excitation, activation, rew_grf, trunk_vel, meta_cost, obs_tx = next_state, next_state_ori, info['muscle_force'], info['excitation'], info['activation'], info['rew_dict']['grf'], info['rew_dict']['gaussian_vel'], info['meta_cost'], info['obs_tx']
    

        if done:
            num_episodes += 1
            total_return += episode_return
            state, state_ori, info = env.reset(return_info=True)
            t = 0
            episode_return = 0.0
            muscle_force, excitation, activation, rew_grf, trunk_vel, meta_cost, obs_tx = info['muscle_force'], info['excitation'], info['activation'], info['rew_dict']['grf'], info['rew_dict']['gaussian_vel'], info['meta_cost'], info['obs_tx']
    
        if switch and t_total % switch_interval == 0:
            print(f'Now is the {switch_cnt} turn')
            print(f't_total {t_total}')
            print(f'Return of the expert is {episode_return}')
            switch_cnt += 1
            if switch_cnt % 2 == 1:
                state, state_ori, info = env.reset(return_info=True, brute_switch=True)
            else:
                state, state_ori, info = env.reset(return_info=True, brute_switch=False) # Here when collecting the expert, DO manually set the env "leg_switch" as False
            t = 0
            episode_return = 0.0
            muscle_force, excitation, activation, rew_grf, trunk_vel, meta_cost, obs_tx = info['muscle_force'], info['excitation'], info['activation'], info['rew_dict']['grf'], info['rew_dict']['gaussian_vel'], info['meta_cost'], info['obs_tx']
    
    if num_episodes == 0:
        num_episodes += 1
        total_return += episode_return

    print(f'Mean return of the expert is {total_return / num_episodes}')
    print(f'Number of episodes is {num_episodes}')
    return buffer



def adjust_demo(buffer_path, new_freq, ori_freq):

    buffer_ori = SerializedBuffer(
        path=buffer_path,
        device=torch.device("cpu")
    )

    
    new_freq = new_freq
    ori_freq = ori_freq

    states_old = buffer_ori.states.clone().numpy()
    next_states_old = buffer_ori.next_states.clone().numpy()
    muscle_forces_old = buffer_ori.muscle_forces.clone().numpy()
    excitations_old = buffer_ori.excitations.clone().numpy()
    activations_old = buffer_ori.activations.clone().numpy()
    rew_grfs_old = buffer_ori.rew_grfs.clone().numpy()
    trunk_vels_old = buffer_ori.trunk_vels.clone().numpy()
    meta_costs_old = buffer_ori.meta_costs.clone().numpy()
    obs_tx_old = buffer_ori.obs_tx.clone().numpy()

    ori_time = np.linspace(0, states_old.shape[0], states_old.shape[0])
    new_time = np.linspace(ori_time[0], ori_time[-1], int(len(ori_time) * (new_freq / ori_freq)))

    states_new = np.zeros((int(len(ori_time) * (new_freq / ori_freq)), buffer_ori.states.size(1)))
    next_states_new = np.zeros((int(len(ori_time) * (new_freq / ori_freq)), buffer_ori.states.size(1)))
    muscle_forces_new = np.zeros((int(len(ori_time) * (new_freq / ori_freq)), buffer_ori.muscle_forces.size(1)))
    excitations_new = np.zeros((int(len(ori_time) * (new_freq / ori_freq)), buffer_ori.muscle_forces.size(1)))
    activations_new = np.zeros((int(len(ori_time) * (new_freq / ori_freq)), buffer_ori.muscle_forces.size(1)))
    rew_grfs_new = np.zeros((int(len(ori_time) * (new_freq / ori_freq)), buffer_ori.rew_grfs.size(1)))
    trunk_vels_new = np.zeros((int(len(ori_time) * (new_freq / ori_freq)), buffer_ori.trunk_vels.size(1)))
    meta_costs_new = np.zeros((int(len(ori_time) * (new_freq / ori_freq)), buffer_ori.meta_costs.size(1)))
    obs_tx_new = np.zeros((int(len(ori_time) * (new_freq / ori_freq)), buffer_ori.obs_tx.size(1)))


    buffer = Buffer(
        buffer_size=int(len(ori_time) * (new_freq / ori_freq)),
        state_shape=[buffer_ori.states.size(1)],
        action_shape=[buffer_ori.actions.size(1)],
        muscle_shape=buffer_ori.muscle_forces.size(1), 
        device=torch.device("cpu")
    )
    
    # Adjust states and next_states
    for i in tqdm(range(states_old.shape[1])):
        interpolator_states = interpolate.CubicSpline(ori_time, states_old[:, i])
        states_new[:, i] = interpolator_states(new_time)[:states_new.shape[0]]

        interpolator_next_states = interpolate.CubicSpline(ori_time, next_states_old[:, i])
        next_states_new[:, i] = interpolator_next_states(new_time)[:next_states_new.shape[0]]
    
    # Adjust muscle forces, excitation and activation
    for i in tqdm(range(muscle_forces_old.shape[1])):
        interpolator_muscle_forces= interpolate.CubicSpline(ori_time, muscle_forces_old[:, i])
        muscle_forces_new[:, i] = interpolator_muscle_forces(new_time)[:muscle_forces_new.shape[0]]

        interpolator_excitations= interpolate.CubicSpline(ori_time, excitations_old[:, i])
        excitations_new[:, i] = interpolator_excitations(new_time)[:excitations_new.shape[0]]

        interpolator_activations= interpolate.CubicSpline(ori_time, activations_old[:, i])
        activations_new[:, i] = interpolator_activations(new_time)[:activations_new.shape[0]]

    # Adjust rew_grf, trunck_vel, meta_cost
    for i in tqdm(range(rew_grfs_old.shape[1])):
        interpolator_rew_grfs= interpolate.CubicSpline(ori_time, rew_grfs_old[:, i])
        rew_grfs_new[:, i] = interpolator_rew_grfs(new_time)[:rew_grfs_new.shape[0]]

        interpolator_trunk_vels= interpolate.CubicSpline(ori_time, trunk_vels_old[:, i])
        trunk_vels_new[:, i] = interpolator_trunk_vels(new_time)[:trunk_vels_new.shape[0]]

        interpolator_meta_costs= interpolate.CubicSpline(ori_time, meta_costs_old[:, i])
        meta_costs_new[:, i] = interpolator_meta_costs(new_time)[:meta_costs_new.shape[0]]

        interpolator_obs_txs= interpolate.CubicSpline(ori_time, obs_tx_old[:, i])
        obs_tx_new[:, i] = interpolator_obs_txs(new_time)[:obs_tx_new.shape[0]]
    
    for i in tqdm(range(states_new.shape[0])):
        actions = np.zeros(buffer_ori.actions.size(1))
        done = False
        rewards = 0
        # Only states and next_states have meaning
        buffer.append(states_new[i], actions, rewards, done, next_states_new[i], muscle_forces_new[i], excitations_new[i], activations_new[i], rew_grfs_new[i], trunk_vels_new[i], meta_costs_new[i], obs_tx_new[i])
    
    
    return buffer


    