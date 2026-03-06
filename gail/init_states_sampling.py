import os
import argparse
from datetime import datetime
import torch
import gym
import sconegym
import numpy as np
import pdb
from sconetools import sconepy
import matplotlib.pyplot as plt
from tqdm import tqdm

from gail_demo.buffer import SerializedBuffer

def run_simulation(model, store_data, max_time=10, min_com_height=0.9, sample_rate=100.0, dof_pos=None, dof_vel=None, activation=None, cnt=None):
    """Run a simulation with a SCONE model.

    Args:
        model: SCONE model
        store_data: Store simulation results
        max_time: Maximum simulation time
        min_com_height: Minimum height of the model center of mass"""
    

    time = []
    effort = []
    trajectories = []
    grf = []
    inputs = []
    sucess = True


    model.set_store_data(True)
 
    model.set_dof_positions(dof_pos) 
    model.set_dof_velocities(dof_vel)

    # model.init_muscle_activations(activation - model.muscle_activation_array())
    model.init_state_from_dofs()


    # Simulation loop
    for t in np.arange(0, max_time, 1.0 / sample_rate):
        time.append(t)

        current_effort = model.measure().current_result(model)
        effort.append(current_effort)
        # print(f"t={t:.3f}, Effort: {current_effort}", end='\r', flush=True)

        trajectories.append([model.dofs()[i].pos() for i in range(len(model.dofs()))])
        # pdb.set_trace()
        grf.append(
            [
                model.legs()[0].contact_force().array(),
                model.legs()[1].contact_force().array(),
            ]
        )


        control_input = np.zeros(4)
        input_array = np.zeros(len(model.actuators()))
        input_array[-4:] = control_input
        model.set_actuator_inputs(input_array)


        # Advance the simulation to time t
        model.advance_simulation_to(t)


        # Abort the simulation if the model center of mass falls below 0.3 meter
        com_y = model.com_pos().y
        # print(com_y)
        if com_y < min_com_height:
            #print(f"Aborting simulation at t={model.time():.3f} com_y={com_y:.4f}")
            sucess = False
            # pdb.set_trace()
            break

    total_effort = model.measure().final_result(model)

    # Store results
    if sucess:
        
        dirname = "init70hamstringweakness_v1" + model.name()
        filename = f'{model.name()}_{cnt}' + ""  # "_baseline", + "_disturbed"
        model.write_results(dirname, filename)
        print(
            f"Results written to {dirname}/{filename}",
            flush=True,
        )

    return (
        time,
        trajectories,
        grf,
        effort,
        total_effort,
        inputs,
        sucess
    )

def init_values(env, buffer):
    store_data = False
    max_sim_time = 10.0
    sample_rate = 100.0

    positive = 0
    negative = 0
    init = []

    for i in tqdm(range(3000)):
        env.reset()
        model = env.model
        dof_pos = buffer.states[i][:buffer.states.shape[1] // 2].cpu().numpy()
        dof_vel = buffer.states[i][buffer.states.shape[1] // 2 : buffer.states.shape[1]].cpu().numpy()
        activation = buffer.activations[i]


        time, trajectories, grf, effort, total_effort, inputs, sucess = run_simulation(
            model, store_data, max_time=max_sim_time, sample_rate=sample_rate, dof_pos=dof_pos, dof_vel=dof_vel, activation=activation, cnt=i
        )

        #print('current loop: ', i, sucess)#, end='\r', flush=True)
        # pdb.set_trace()
        if sucess:
            positive += 1
            init.append(i)
        else:
            negative += 1

    print(f'positive: {positive} {positive / (positive + negative)} negative: {negative} {negative / (positive + negative)}')
    # print(init)
    

    return init



def main():
    """Example of running a simulation with a SCONE model."""
    # Set the SCONE log level between 1-7 (lower is more logging)
    sconepy.set_log_level(1)
    print("SCONE Version", sconepy.version())
    print()
    sconepy.set_array_dtype_float32()




    # env_exp = gym.make('sconewalk_origin_motored_normal_h0914-v3')
    # env_shortham = gym.make('sconewalk_origin_motored_shorthamstring_h0914-v3')
    env_hamweak = gym.make('sconewalk_origin_motored_hamstringweakness_new_h0914-v3')
    # env_plantarweak = gym.make('sconewalk_origin_motored_plantarweakness_h0914-v3')
    # env_iliopsoasweak = gym.make('sconewalk_origin_motored_iliopsoasweakness_h0914-v3')
    # buffer_exp = SerializedBuffer(path="buffers/sconewalk_origin_motored_normal_h0914-v3/v1/Adjusted_Same70Init_dof_size1000000_for_hamstringweakness_new_freq1.43_100hz.pth", device="cpu")
    buffer_exp = SerializedBuffer(path="buffers/sconewalk_origin_motored_normal_h0914-v3/v2/Adjusted_Same70Init_dof_size1144000_for_hamstringweakness_new_freq1.43_100hz.pth", device="cpu")
    buffer_exp = SerializedBuffer(path="buffers/sconewalk_origin_motored_normal_h0914-v3/v2/Adjusted_Same70Init_dof_size888000_for_iliopsoasweakness_new_freq1.11_100hz.pth", device="cpu")
    buffer_exp = SerializedBuffer(path="buffers/sconewalk_origin_motored_normal_h0914-v3/v2/Adjusted_Same70Init_dof_size888000_for_iliopsoasweakness_new_freq1.11_100hz.pth", device="cpu")
    
    pdb.set_trace()
    # init_exp = init_values(env_exp, buffer_exp)
    # init_shortham = init_values(env_shortham, buffer_exp)
    init_hamweak = init_values(env_hamweak, buffer_exp)
    # init_plantarweak = init_values(env_plantarweak, buffer_exp)
    # init_iliopsoasweak = init_values(env_iliopsoasweak, buffer_exp)

    # init = [x for x in init_iliopsoasweak if x in init_exp]
    # print(init_exp)
    print(init_hamweak)
    # print(init)



    






if __name__ == "__main__":
    main()





