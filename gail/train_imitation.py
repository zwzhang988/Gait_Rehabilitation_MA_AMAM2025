import os
import argparse
from argparse import Namespace
from datetime import datetime
import torch
import gym
import sconegym
import wandb
import pdb

#from gail_airl_ppo.env import make_env
from gail_demo.buffer import SerializedBuffer
from gail_demo.algo import ALGOS
from gail_demo.trainer import Trainer


def run(args):
    env_hamwl = gym.make('sconewalk_origin_motored_hamstringweakness_new_h0914-v3')
    env_test_hamwl = gym.make('sconewalk_origin_motored_hamstringweakness_new_h0914-v3')
    env_iliowl = gym.make('sconewalk_origin_motored_iliopsoasweakness_h0914-v3')
    env_test_iliowl = gym.make('sconewalk_origin_motored_iliopsoasweakness_h0914-v3')
    env_shortham = gym.make('sconewalk_origin_motored_shorthamstring_h0914-v3')
    env_test_shortham = gym.make('sconewalk_origin_motored_shorthamstring_h0914-v3')
    env = {'sconewalk_origin_motored_hamstringweakness_new_h0914-v3': env_hamwl,
           'sconewalk_origin_motored_iliopsoasweakness_h0914-v3': env_iliowl, 
           'sconewalk_origin_motored_shorthamstring_h0914-v3': env_shortham, }
    env_test = {'sconewalk_origin_motored_hamstringweakness_new_h0914-v3': env_test_hamwl, 
                'sconewalk_origin_motored_iliopsoasweakness_h0914-v3': env_test_iliowl, 
                'sconewalk_origin_motored_shorthamstring_h0914-v3': env_test_shortham, }

    buffer_exp_hamwl = SerializedBuffer(
        path=os.path.expanduser("~") + '/MA/codes/gail/buffers/sconewalk_origin_motored_normal_h0914-v3/v2/Adjusted_Same70Init_dof_size1144000_for_hamstringweakness_new_freq1.43_100hz.pth',
        device=torch.device("cuda" if args.cuda else "cpu")
    )
    buffer_exp_iliowl = SerializedBuffer(
        path=os.path.expanduser("~") + '/MA/codes/gail/buffers/sconewalk_origin_motored_normal_h0914-v3/v2/Adjusted_Same70Init_dof_size888000_for_iliopsoasweakness_new_freq1.11_100hz.pth',
        device=torch.device("cuda" if args.cuda else "cpu")
    )
    buffer_exp_shortham = SerializedBuffer(
        path=os.path.expanduser("~") + '/MA/codes/gail/buffers/sconewalk_origin_motored_normal_h0914-v3/v2/Adjusted_Same70Init_dof_size888000_for_shorthamstring_new_freq1.11_100hz.pth',
        device=torch.device("cuda" if args.cuda else "cpu")
    )
    buffer_exp = {'sconewalk_origin_motored_hamstringweakness_new_h0914-v3': buffer_exp_hamwl, 
                  'sconewalk_origin_motored_iliopsoasweakness_h0914-v3': buffer_exp_iliowl, 
                  'sconewalk_origin_motored_shorthamstring_h0914-v3': buffer_exp_shortham, }
    # buffer_exp = SerializedBuffer(path=args.buffer, device='cuda:0')
    # pdb.set_trace()
    # tmp = buffer_exp.states[3:,:].clone()
    # buffer_exp.states[0:-3, :] = tmp.clone()
    # tmp = buffer_exp.next_states[3:,:].clone()
    # buffer_exp.next_states[0:-3, :] = tmp.clone()

    header_num = 3
    obs_history = 1
    act_chunked = 4

    algo = ALGOS[args.algo](
        buffer_exp=buffer_exp,
        state_shape=env_hamwl.observation_space.shape,
        action_shape=env_hamwl.action_space.shape,
        muscle_shape=len(env_hamwl.model.muscles()), 
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        rollout_length=args.rollout_length, 
        units_actor=(512, 512), 
        units_critic=(512, 512), 
        #PPO (512, 512) 
        units_disc=(64, 64), 
        header_num=header_num, 
        obs_history=obs_history,
        act_chunked=act_chunked, 
        num_steps=args.num_steps,
    )
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', 'None', args.algo, f'seed{args.seed}-{time}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed, 
        draw=False, 
        # agent_env_id=args.env_id, 
        # expert_env_id=args.expert_env_id, 
        max_step=10**4, 
        device=torch.device("cuda" if args.cuda else "cpu"),
        animating=False, 
        header_num=header_num, 
        obs_history=obs_history,
        act_chunked=act_chunked,  
    )
    trainer.train()

    # dissimilarity, objective = trainer.train()

# def run():
#     wandb.init(project="test_sweep_2")
#     config = wandb.config
#     env = gym.make(config.env_id)
#     env_test = gym.make(config.env_id)
#     buffer_exp = SerializedBuffer(
#         path=config.buffer,
#         device=torch.device("cuda" if config.cuda else "cpu")
#     )

#     algo = ALGOS[config.algo](
#         buffer_exp=buffer_exp,
#         state_shape=env.observation_space.shape,
#         action_shape=env.action_space.shape,
#         muscle_shape=len(env.model.muscles()), 
#         device=torch.device("cuda" if config.cuda else "cpu"),
#         seed=config.seed,
#         rollout_length=config.rollout_length, 
#         units_actor=(512, 512), 
#         units_critic=(512, 512), 
#         #PPO (512, 512) 
#         units_disc=(64, 64), 
#     )

#     time = datetime.now().strftime("%Y%m%d-%H%M")
#     log_dir = os.path.join(
#         'logs', config.env_id, config.algo, f'seed{config.seed}-{time}')

#     trainer = Trainer(
#         env=env,
#         env_test=env_test,
#         algo=algo,
#         log_dir=log_dir,
#         num_steps=config.num_steps,
#         eval_interval=config.eval_interval,
#         seed=config.seed, 
#         draw=False, 
#         agent_env_id=config.env_id, 
#         expert_env_id=config.expert_env_id, 
#         max_step=10**4
#     )

    
#     # pdb.set_trace()

    

#     dissimilarity, objective = trainer.train(config=wandb.config)
#     wandb.log({'dissimilarity': dissimilarity})
#     wandb.log({'objective': objective})
    


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # p.add_argument('--buffer', type=str, required=True)
    p.add_argument('--rollout_length', type=int, default=50000)
    p.add_argument('--num_steps', type=int, default=10**7)
    p.add_argument('--eval_interval', type=int, default=10**5)
    # p.add_argument('--env_id', type=str, default='Hopper-v3')
    # p.add_argument('--expert_env_id', type=str, default='Hopper-v3')
    p.add_argument('--algo', type=str, default='gail')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--coeff_gaussian_vel', type=float, default=1)
    p.add_argument('--coeff_rew_grf', type=float, default=1)
    p.add_argument('--coeff_rew_constr', type=float, default=1)
    p.add_argument('--coeff_disc_rewards', type=float, default=1)
    args = p.parse_args()
    run(args)

# sweep_config = {
#     "method": "bayes", # 'random'
#     "metric": {"goal": "minimize", "name": "objective"}, 
#     "parameters": {
#         'buffer': {'value': 'buffers/sconewalk_origin_motored_normal_h0914-v3/v1/Adjusted_Same70Init_dof_size1000000_for_shorthamstring_new_freq1.11_100hz.pth'}, 
#         'rollout_length': {'value': 100000}, #100000
#         'num_steps': {'value': 50000000}, #50000000
#         'eval_interval': {'value': 5000000}, #5000000
#         'env_id': {'value': 'sconewalk_origin_motored_shorthamstring_h0914-v3'}, 
#         'expert_env_id': {'value': 'sconewalk_origin_motored_normal_h0914-v3'}, 
#         'algo': {'value': 'gail'}, 
#         'cuda': {'value': True}, 
#         'seed': {'value': 0}, 
#         "coeff_gaussian_vel": {"values" : [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]}, 
#         "coeff_rew_grf": {"values" : [-3.0, -2.0, -1.0, -0.5, -0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0]}, 
#         "coeff_rew_constr": {"values" : [-3.0, -2.0, -1.0, -0.5, -0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0]}, 
#         "coeff_disc_rewards": {"values" : [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]}, 
#     }
# }


# # coeff_gaussian_vel: {"values" : [-3.0, -1.0, ..., 0.0, 0.1, 0.5, 1.0, 3.0, 10.0, 50.0, 100.0]}

# sweep_id = wandb.sweep(sweep=sweep_config, project="test_sweep_2")

# wandb.agent(sweep_id, function=run, count=12)


