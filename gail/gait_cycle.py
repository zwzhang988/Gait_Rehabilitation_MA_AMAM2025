import numpy as np
from scipy.interpolate import interp1d
import pdb


def determine_gait_cycle(time, grf):
    """Determine the gait cycle times from the ground reaction force data."""
    gait_cycle_times_l = []
    gait_cycle_times_r = []
    #pdb.set_trace()

    for i in range(1, len(grf)):
        if grf[i][0][1] > 0 and grf[i - 1][0][1] < 1e-10:
            gait_cycle_times_l.append(time[i])
        if grf[i][1][1] > 0 and grf[i - 1][1][1] < 1e-10:
            gait_cycle_times_r.append(time[i])

    return gait_cycle_times_l, gait_cycle_times_r


def transform_time_to_gait_cycle(time, gait_cycle_times, trajectories, grf, num_points=100):
    """Transform the time series data to gait cycle data."""
    gait_cycle_trajectories = []
    gait_cycle_grf = []
    #pdb.set_trace()

    for i in range(len(gait_cycle_times) - 1):
        gait_cycle_start_time = gait_cycle_times[i]
        gait_cycle_end_time = gait_cycle_times[i + 1]

        gait_cycle_start_idx = time.index(gait_cycle_start_time)
        gait_cycle_end_idx = time.index(gait_cycle_end_time)

        # Interpolate for trajectories
        time_slice = time[gait_cycle_start_idx:gait_cycle_end_idx]
        traj_slice = np.array(trajectories[gait_cycle_start_idx:gait_cycle_end_idx])
        interp_func = interp1d(time_slice, traj_slice, axis=0, kind="linear")
        interpolated_time = np.linspace(time_slice[0], time_slice[-1], num_points)
        interpolated_traj = interp_func(interpolated_time)

        # Interpolate for grf
        grf_slice = np.array(grf[gait_cycle_start_idx:gait_cycle_end_idx])
        interp_func_grf = interp1d(time_slice, grf_slice, axis=0, kind="linear")
        interpolated_grf = interp_func_grf(interpolated_time)

        gait_cycle_trajectories.append(interpolated_traj)
        gait_cycle_grf.append(interpolated_grf)

    return gait_cycle_trajectories, gait_cycle_grf
