import numpy as np
import matplotlib.pyplot as plt


def plot_simulation_in_time(time, trajectories, grf, gait_cycle_times, inputs, model):
    """Plot the simulation data/trajectories over time."""
    # Assuming 'dofs' and 'trajectories' are defined
    # Create a subplot for each selected joint
    selected_joints = [
        "hip_flexion_r",
        # "hip_flexion_l",
        "knee_angle_r",
        # "knee_angle_l",
        "ankle_angle_r",
        # "ankle_angle_l",
    ]

    # Count the number of selected joints
    num_selected_joints = len(selected_joints)

    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(num_selected_joints + 2, 1, figsize=(10, (num_selected_joints + 2) * 3))

    # Iterate through each degree of freedom
    subplot_idx = 0
    dofs = model.dofs()
    for i in range(len(dofs)):
        if dofs[i].name() in selected_joints:
            ax = axs[subplot_idx]
            ax.plot(
                time,
                [t[i] for t in trajectories],
                label=dofs[i].name(),
            )
            for gait_cycle_time in gait_cycle_times:
                ax.axvline(gait_cycle_time, color="r", linestyle="--")
            # ax.set_title(f"Trajectory of {dofs[i].name()}")
            # ax.set_xlabel("Time [s]")
            ax.set_ylabel(f"{dofs[i].name()} [rad]")
            ax.grid()
            # ax.legend()
            subplot_idx += 1

    ax = axs[-2]
    ax.plot(time, [np.linalg.norm(grf[j][0]) for j in range(0, len(grf))])
    for gait_cycle_time in gait_cycle_times:
        ax.axvline(gait_cycle_time, color="r", linestyle="--")
    ax.set_ylabel("L2 of GRF [N]")
    ax.grid()

    ax = axs[-1]
    ax.plot(time, [i[0] for i in inputs], label="Input 1")
    ax.plot(time, [i[1] for i in inputs], label="Input 2")
    for gait_cycle_time in gait_cycle_times:
        ax.axvline(gait_cycle_time, color="r", linestyle="--")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Input")
    ax.grid()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.draw()


def plot_simulation_in_gait_cycle(trajectories, grf, model):
    """Plot the simulation data/trajectories over the gait cycle."""
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
