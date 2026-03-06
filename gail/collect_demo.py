import os
import sys
import argparse
import torch
import sconegym
import gym
import pdb

#from gail_airl_ppo.env import make_env
from gail_demo.algo import SACExpert
from gail_demo.utils import collect_demo


def run(args):
    env = gym.make(args.env_id)
    '''
    algo = SACExpert(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        path=args.weight
    )
    '''

    buffer = collect_demo(
        env=env,
        algo=None,
        buffer_size=args.buffer_size,
        device=torch.device("cuda" if args.cuda else "cpu"),
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed, 
        switch=args.switch, 
        switch_interval=args.switch_interval, 
    )
    buffer.save(os.path.join(
        'buffers',
        args.env_id,
        'v2',
        f'Same70Init_dof_size{args.buffer_size}_std{args.std}_prand{args.p_rand}_100hz.pth'
    ))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weight', type=str, required=False)
    p.add_argument('--env_id', type=str, default='Hopper-v3')
    p.add_argument('--buffer_size', type=int, default=10**6)
    p.add_argument('--std', type=float, default=0.0)
    p.add_argument('--p_rand', type=float, default=0.0)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--switch', action='store_true')
    p.add_argument('--switch_interval', type=int, default=500000)
    args = p.parse_args()
    run(args)
