import sys
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gym
from sconetools import sconepy

from gait_cycle import determine_gait_cycle, transform_time_to_gait_cycle
from frequency import plot_simulation_in_time, plot_simulation_in_gait_cycle

import pdb

def model_info(model):
    print()
    print("Model info:")

    actuators = model.actuators()
    print("Actuators: ", end="")
    for a in actuators:
        print(a.name(), end=", ")
    print()

    bodies = model.bodies()
    print("Bodies: ", end="")
    for b in bodies:
        print(b.name(), end=", ")
    print()

    joints = model.joints()
    print("Joints: ", end="")
    for j in joints:
        print(j.name(), end=", ")
    print()

    dofs = model.dofs()
    print("DOFs: ", end="")
    for d in dofs:
        print(d.name(), end=", ")
    print()

    measures = model.measure()
    print("Measure: ", end="")
    print(measures.name())

    print()


def run_simulation(model, store_data, max_time=5, min_com_height=0.3, sample_rate=100.0, controller=None):
    """Run a simulation with a SCONE model.

    Args:
        model: SCONE model
        store_data: Store simulation results
        max_time: Maximum simulation time
        min_com_height: Minimum height of the model center of mass"""
    model.set_store_data(store_data)

    time = []
    effort = []
    trajectories = []
    grf = []
    inputs = []

    # Simulation loop
    for t in np.arange(0, max_time, 1.0 / sample_rate):
        time.append(t)

        current_effort = model.measure().current_result(model)
        effort.append(current_effort)
        print(f"t={t:.3f}, Effort: {current_effort}", end="\r", flush=True)

        trajectories.append([model.dofs()[i].pos() for i in range(len(model.dofs()))])
        # pdb.set_trace()
        grf.append(
            [
                model.legs()[0].contact_force().array(),
                model.legs()[1].contact_force().array(),
            ]
        )

        # Apply control inputs
        if controller is None:
            control_input = np.zeros(4)
            inputs.append(control_input)
            input_array = np.zeros(len(model.actuators()))
            input_array[-4:] = control_input
            model.set_actuator_inputs(input_array)
        else:
            control_input = controller(model, t)
            inputs.append(control_input)
            input_array = np.zeros(len(model.actuators()))
            input_array[-4:] = control_input
            #input_array[:18] = np.ones(18) * 1
            model.set_actuator_inputs(input_array)
            #pdb.set_trace()
            #print(model.muscle_fiber_length_array())

        # Advance the simulation to time t
        model.advance_simulation_to(t)

        # Abort the simulation if the model center of mass falls below 0.3 meter
        com_y = model.com_pos().y
        if com_y < min_com_height:
            print(f"Aborting simulation at t={model.time():.3f} com_y={com_y:.4f}")
            break

    total_effort = model.measure().final_result(model)

    # Store results
    if store_data:
        dirname = "sconepy_reflex_gait_scenario_actuated" + model.name()
        filename = model.name() + "_controlled"  # "_baseline", + "_disturbed"
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
    )


def simulate(env_id):
    """Example of running a simulation with a SCONE model."""
    # Set the SCONE log level between 1-7 (lower is more logging)
    sconepy.set_log_level(1)
    print("SCONE Version", sconepy.version())
    print()
    sconepy.set_array_dtype_float32()

    store_data = True
    max_sim_time = 20.0
    #pdb.set_trace()

    env = gym.make(env_id)
    model = env.model

    # if sconepy.is_supported("ModelHyfydy"):
    #     model = sconepy.load_model("scone_scenarios/reflex_gait_H0918Gait/Hy0918v3_motored.scone")
    #     model = sconepy.load_model("scone_scenarios/H0914_limited_prove/H0914_hamstring_limited_test.scone")
    #     model = sconepy.load_model("scone_scenarios/H0914_hamstring_origin_prove/H0914_hamstring_origin_test.scone")
    #     model = sconepy.load_model("scone_scenarios/H0914_normal_origin_prove/H0914_normal_origin_test.scone")
    #     pdb.set_trace()
    # if sconepy.is_supported("ModelOpenSim3"):
    #    model = sconepy.load_model("scone_scenarios/reflex_gait_H0918Gait/reflex_gait_scenario.scone")

    model_info(model)

    controller = None
    sample_rate = 100.0

    time, trajectories, grf, effort, total_effort, inputs = run_simulation(
        model, store_data, max_time=max_sim_time, sample_rate=sample_rate,controller=controller
    )
    print("Total effort:", total_effort)
    #pdb.set_trace()

    gait_cycle_times_l, gait_cycle_times_r = determine_gait_cycle(time, grf)

    # Transform data to gait cycle using left leg
    trajectories_gait_cycle, grf_gait_cylce = transform_time_to_gait_cycle(
        time, gait_cycle_times_l, trajectories, grf
    )

    # Pickle simulation data
    # Chek if the directory exists
    # if not os.path.exists("simulation_data"):
    #     os.makedirs("simulation_data")
    # with open("simulation_data/simulation_data.pkl", "wb") as f:
    #     pickle.dump(
    #         {
    #             "time": time,
    #             "trajectories": trajectories,
    #             "grf": grf,
    #             "trajectories_gait_cycle": trajectories_gait_cycle,
    #             "grf_gait_cylce": grf_gait_cylce,
    #             "effort": effort,
    #             "total_effort": total_effort,
    #             "gait_cycle_times_l": gait_cycle_times_l,
    #             "gait_cycle_times_r": gait_cycle_times_r,
    #             "inputs": inputs,
    #         },
    #         f,
    #     )

    # Plotting
    plot_simulation_in_time(time, sample_rate, trajectories, grf, gait_cycle_times_l, gait_cycle_times_r, inputs, model, trained_steps=0)
    # plot_simulation_in_gait_cycle(trajectories_gait_cycle, grf_gait_cylce, model)
    # plt.show()


if __name__ == "__main__":
    simulate()
