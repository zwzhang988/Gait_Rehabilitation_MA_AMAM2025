import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import wandb
import pdb

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

def plot_simulation_in_time(time, sample_rate, trajectories, grf, gait_cycle_times_l, gait_cycle_times_r, actions, model, trained_steps):
    """Plot the simulation data/trajectories over time."""
    
    # Assuming 'dofs' and 'trajectories' are defined
    # Create a subplot for each selected joint
    selected_joints_r = [
        "hip_flexion_r",
        # "hip_flexion_l",
        "knee_angle_r",
        # "knee_angle_l",
        "ankle_angle_r",
        # "ankle_angle_l",
    ]
    selected_joints_l = [
        # "hip_flexion_r",
        "hip_flexion_l",
        # "knee_angle_r",
        "knee_angle_l",
        # "ankle_angle_r",
        "ankle_angle_l",
    ]

    # Count the number of selected joints
    num_selected_joints_r = len(selected_joints_r)

    # Create a figure and a grid of subplots
    fig_r, axs_r = plt.subplots(num_selected_joints_r * 2, 1, figsize=(10, (num_selected_joints_r * 2) * 3))
    fig_r.suptitle(f'Joint rotation frequncies Right {trained_steps} steps')
    fig_l, axs_l = plt.subplots(num_selected_joints_r * 2, 1, figsize=(10, (num_selected_joints_r * 2) * 3))
    fig_l.suptitle(f'Joint rotation frequncies Left {trained_steps} steps')

    # Iterate through each degree of freedom
    subplot_idx_r = 0
    subplot_idx_l = 0

    dofs = model.dofs()
    for i in range(len(dofs)):
        if dofs[i].name() in selected_joints_r:
            ax = axs_r[subplot_idx_r]
            
            series = [t[i] for t in trajectories]

            ax.plot(
                time,
                series,
                label=dofs[i].name(),
            )
            for gait_cycle_time in gait_cycle_times_r:
                ax.axvline(gait_cycle_time, color="r", linestyle="--")
            # ax.set_title(f"Trajectory of {dofs[i].name()}")
            # ax.set_xlabel("Time [s]")
            ax.set_ylabel(f"{dofs[i].name()} [rad]")
            ax.grid()
            # ax.legend()

            ax = axs_r[subplot_idx_r + 1]

            # seriesf = fft(series)
            # xf = fftfreq(len(time), 1.0 / sample_rate)
            # seriesf_pos = 2.0/len(time) * np.abs(seriesf[0:len(time) // 2])
            # xf_pos = xf[:len(time) // 2]

            # dominant_omega, peaks = find_dominant_freq(xf_pos, seriesf_pos)
            '''
            xf_pos, seriesf_pos, dominant_omega, peaks = freq_analysis(series, len(time), sample_rate)
            dominant_freq = 1.0 / dominant_omega

            ax.plot(xf_pos, seriesf_pos)
            ax.plot(xf_pos[peaks], seriesf_pos[peaks], "x")
            # ax.plot(xf, seriesf)
            # ax.plot(xf[peaks], seriesf[peaks], "x")
            ax.set_ylabel(f"{dominant_freq[0]:.3f} [Hz]")
            ax.grid()
            '''

            subplot_idx_r += 2

        if dofs[i].name() in selected_joints_l:
            ax = axs_l[subplot_idx_l]
            
            series = [t[i] for t in trajectories]

            ax.plot(
                time,
                series,
                label=dofs[i].name(),
            )
            for gait_cycle_time in gait_cycle_times_l:
                ax.axvline(gait_cycle_time, color="r", linestyle="--")
            # ax.set_title(f"Trajectory of {dofs[i].name()}")
            # ax.set_xlabel("Time [s]")
            ax.set_ylabel(f"{dofs[i].name()} [rad]")
            ax.grid()
            # ax.legend()

            ax = axs_l[subplot_idx_l + 1]
            # seriesf = fft(series)
            # xf = fftfreq(len(time), 1.0 / sample_rate)
            # seriesf_pos = 2.0/len(time) * np.abs(seriesf[0:len(time) // 2])
            # xf_pos = xf[:len(time) // 2]

            # dominant_omega, peaks = find_dominant_freq(xf_pos, seriesf_pos)
            '''
            xf_pos, seriesf_pos, dominant_omega, peaks = freq_analysis(series, len(time), sample_rate)
            dominant_freq = 1.0 / dominant_omega

            ax.plot(xf_pos, seriesf_pos)
            ax.plot(xf_pos[peaks], seriesf_pos[peaks], "x")
            # ax.plot(xf, seriesf)
            # ax.plot(xf[peaks], seriesf[peaks], "x")
            ax.set_ylabel(f"{dominant_freq[0]:.3f} [Hz]")
            ax.grid()
            '''

            subplot_idx_l += 2

    # pdb.set_trace()

    # Plot for ground touching forces
    fig_grf, axs_grf = plt.subplots(2 * 2, 1, figsize=(10, (2 * 2) * 3))
    fig_grf.suptitle(f'Ground touching forces {trained_steps} steps')
    ax = axs_grf[0]
    series = [np.linalg.norm(grf[j][0]) for j in range(0, len(grf))]
    ax.plot(time, series)
    for gait_cycle_time in gait_cycle_times_l:
        ax.axvline(gait_cycle_time, color="r", linestyle="--")
    ax.set_ylabel("L2 of GRF_Left [N]")
    ax.grid()

    '''
    xf_pos, seriesf_pos, dominant_omega, peaks = freq_analysis(series, len(time), sample_rate)
    dominant_freq = 1.0 / dominant_omega
    ax = axs_grf[1]
    ax.plot(xf_pos, seriesf_pos)
    ax.plot(xf_pos[peaks], seriesf_pos[peaks], "x")
    ax.set_ylabel(f"{dominant_freq[0]:.3f} [Hz]")
    ax.grid()
    '''

    ax = axs_grf[2]
    series = [np.linalg.norm(grf[j][1]) for j in range(0, len(grf))]
    ax.plot(time, series)
    for gait_cycle_time in gait_cycle_times_r:
        ax.axvline(gait_cycle_time, color="r", linestyle="--")
    ax.set_ylabel("L2 of GRF_Right [N]")
    ax.grid()
    
    '''
    xf_pos, seriesf_pos, dominant_omega, peaks = freq_analysis(series, len(time), sample_rate)
    dominant_freq = 1.0 / dominant_omega
    ax = axs_grf[3]
    ax.plot(xf_pos, seriesf_pos)
    ax.plot(xf_pos[peaks], seriesf_pos[peaks], "x")
    ax.set_ylabel(f"{dominant_freq[0]:.3f} [Hz]")
    ax.grid()
    '''
    # ax = axs[-2]
    # ax.plot(time, [np.linalg.norm(grf[j][0]) for j in range(0, len(grf))])
    # for gait_cycle_time in gait_cycle_times:
    #     ax.axvline(gait_cycle_time, color="r", linestyle="--")
    # ax.set_ylabel("L2 of GRF [N]")
    # ax.grid()


    # Plot for actions
    actions = np.tanh(actions) * 50.0
    fig_act, axs_act = plt.subplots(2 * 2, 1, figsize=(10, (2 * 2) * 3))
    fig_act.suptitle(f'Actions torques {trained_steps} steps')
    ax = axs_act[0]
    ax.plot(time, [i[0] for i in actions])
    for gait_cycle_time in gait_cycle_times_r:
        ax.axvline(gait_cycle_time, color="r", linestyle="--")
    ax.set_ylabel("Hip_Right [N*m]")
    ax.grid()

    ax = axs_act[1]
    ax.plot(time, [i[1] for i in actions])
    for gait_cycle_time in gait_cycle_times_r:
        ax.axvline(gait_cycle_time, color="r", linestyle="--")
    ax.set_ylabel("Knee_Right [N*m]")
    ax.grid()

    ax = axs_act[2]
    ax.plot(time, [i[2] for i in actions])
    for gait_cycle_time in gait_cycle_times_l:
        ax.axvline(gait_cycle_time, color="r", linestyle="--")
    ax.set_ylabel("Hip_Left [N*m]")
    ax.grid()

    ax = axs_act[3]
    ax.plot(time, [i[3] for i in actions])
    for gait_cycle_time in gait_cycle_times_l:
        ax.axvline(gait_cycle_time, color="r", linestyle="--")
    ax.set_ylabel("Knee_Left [N*m]")
    ax.grid()

    # ax = axs[-1]
    # ax.plot(time, [i[0] for i in inputs], label="Input 1")
    # ax.plot(time, [i[1] for i in inputs], label="Input 2")
    # for gait_cycle_time in gait_cycle_times:
    #     ax.axvline(gait_cycle_time, color="r", linestyle="--")
    # ax.set_xlabel("Time [s]")
    # ax.set_ylabel("Input")
    # ax.grid()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.draw()
    # plt.show()

    if trained_steps == 0:
        fig_r.set_size_inches(16, 10)
        fig_l.set_size_inches(16, 10)
        fig_grf.set_size_inches(16, 10)
        wandb.log({"expert_frequncy_r": wandb.Image(fig_r)})
        plt.close(fig_r)
        wandb.log({"expert_frequncy_l": wandb.Image(fig_l)})
        plt.close(fig_l)
        wandb.log({"expert_frequncy_grf": wandb.Image(fig_grf)})
        plt.close(fig_grf)
    else:    
        fig_r.set_size_inches(16, 10)
        fig_l.set_size_inches(16, 10)
        fig_grf.set_size_inches(16, 10)
        fig_act.set_size_inches(16, 10)
        wandb.log({"frequncy_r": wandb.Image(fig_r)})
        plt.close(fig_r)
        wandb.log({"frequncy_l": wandb.Image(fig_l)})
        plt.close(fig_l)
        wandb.log({"frequncy_grf": wandb.Image(fig_grf)})
        plt.close(fig_grf)
        wandb.log({"Actions": wandb.Image(fig_act)})
        plt.close(fig_act)






def plot_simulation_in_gait_cycle(trajectories, grf, model):
    """Plot the simulation data/trajectories over the gait cycle."""
    #pdb.set_trace()
    trajectories = np.array(trajectories)
    grf = np.array(grf)

    # Create a subplot for each selected joint
    selected_joints = [
        "hip_flexion_r",
        # "hip_flexion_l",
        "knee_angle_r",
        # "knee_angle_l",
        "ankle_angle_r",
        # "ankle_angle_l",
    ]
    num_selected_joints = len(selected_joints)

    fig, axs = plt.subplots(num_selected_joints + 1, 1, figsize=(10, (num_selected_joints + 1) * 3))
    subplot_idx = 0
    dofs = model.dofs()
    for i in range(len(dofs)):
        if dofs[i].name() in selected_joints:
            ax = axs[subplot_idx]
            for j in range(len(trajectories)):
                ax.plot(
                    np.linspace(0, 100, len(trajectories[j])),
                    trajectories[j, :, i],
                    color="C0",
                    alpha=0.5,
                    label=dofs[i].name(),
                )
            # ax.set_title(f"Trajectory of {dofs[i].name()}")
            # ax.set_xlabel("Gait cycle [%]")
            ax.set_ylabel(f"{dofs[i].name()} [rad]")
            ax.grid()
            # ax.legend()
            subplot_idx += 1

    grf = np.linalg.norm(grf, axis=3)
    ax = axs[-1]
    for j in range(len(grf)):
        ax.plot(
            np.linspace(0, 100, len(grf[j])),
            grf[j, :, 0],
            color="C0",
            alpha=0.5,
        )
    ax.set_xlabel("Gait cycle [%]")
    ax.set_ylabel("L2 of GRF [N]")
    ax.grid()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.draw()
