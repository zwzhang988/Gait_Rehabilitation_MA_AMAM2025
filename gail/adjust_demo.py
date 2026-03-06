import os
import sys
import argparse
import torch
import sconegym
import gym
import pdb

#from gail_airl_ppo.env import make_env
from gail_demo.algo import SACExpert
from gail_demo.utils import adjust_demo, trans_quer


def run(args):

    buffer = adjust_demo(args.buffer, args.new_freq, args.ori_freq)
    buffer.save(os.path.join(
        'buffers',
        args.env_id,
        'v2',
        f'Adjusted_Same70Init_dof_size{buffer.buffer_size}_for_hamstringweakness_new_freq{args.new_freq}_100hz.pth'
    ))

def trans_run(args): 
    buffer = trans_quer(args.buffer)
    buffer.save(os.path.join(
        'buffers',
        args.env_id,
        'v1',
        f'Quer_Same70Init_dof_size{buffer.buffer_size}_for_shorthamstring_new_freq{args.new_freq}_100hz.pth'
    ))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env_id', type=str, default='Hopper-v3')
    p.add_argument('--buffer', type=str, required=True)
    p.add_argument('--ori_freq', type=float, required=True)
    p.add_argument('--new_freq', type=float, required=True)
    p.add_argument('--quer', action='store_true')
    args = p.parse_args()

    if not args.quer:
        run(args)
    else: 
        trans_run(args)
