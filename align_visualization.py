from math import cos, sin
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import torch
import pdb
import time
from gail.gail_demo.buffer import SerializedBuffer
from celluloid import Camera

class Coordinate:
    def __init__(self, origin=np.array([0, 0]), r=0.1, joint_angles=0.0, offset=0.0):
        self.r = r
        self.origin = origin
        self.offset = offset
        self.joint_angles = joint_angles + self.offset
        self.x_axis = self.r * np.array([np.cos(self.joint_angles), np.sin(self.joint_angles)])
        self.y_axis = self.r * np.array([np.cos(self.joint_angles + np.pi / 2), np.sin(joint_angles + np.pi / 2)])
    def update(self, origin, joint_angles): 
        self.origin = origin
        self.joint_angles = joint_angles + self.offset
        self.x_axis = self.r * np.array([np.cos(self.joint_angles), np.sin(self.joint_angles)])
        self.y_axis = self.r * np.array([np.cos(self.joint_angles + np.pi / 2), np.sin(joint_angles + np.pi / 2)])
    def plot(self): 
        # plt.arrow(self.origin[0], self.origin[1], self.x_axis[0], self.x_axis[1], color='b')
        # plt.arrow(self.origin[0], self.origin[1], self.y_axis[0], self.y_axis[1], color='g')
        plt.plot([self.origin[0], self.origin[0]+self.x_axis[0]], 
                 [self.origin[1], self.origin[1]+self.x_axis[1]], 'r')
        plt.plot([self.origin[0], self.origin[0]+self.y_axis[0]], 
                 [self.origin[1], self.origin[1]+self.y_axis[1]], 'g')


class TwoLinkArm:
    # Same for lower limb, shoulder=hip, elbow=knee, wrist=ankle, 
    def __init__(self, joint_angles_l=[0, 0, 0], joint_angles_r=[0, 0, 0]):
        self.pelvis = np.array([0, 0])
        self.hip = self.pelvis + np.array([0, -0.0351])
        self.link_lengths = [0.396, 0.43] # thigh and shank

        self.pelvis_coordi = Coordinate()
        self.hip_l_coordi = Coordinate()
        self.hip_r_coordi = Coordinate()
        self.knee_l_coordi = Coordinate()
        self.knee_r_coordi = Coordinate()
        self.ankle_l_coordi = Coordinate()
        self.ankle_r_coordi = Coordinate()

        self.update_joints(self.pelvis, joint_angles_l, joint_angles_r)

    def update_joints(self, pelvis_position, joint_angles_l, joint_angles_r): # joint values for [hip, knee, ankle]
        self.pelvis = pelvis_position
        self.hip = self.pelvis + np.array([0, -0.0351])
        self.joint_angles_l = joint_angles_l - np.array([np.pi / 2, 0, 0])
        self.joint_angles_r = joint_angles_r - np.array([np.pi / 2, 0, 0])
        self.forward_kinematics()

        self.pelvis_coordi.update(self.pelvis, 0)
        self.hip_l_coordi.update(self.hip, self.joint_angles_l[0])
        self.hip_r_coordi.update(self.hip, self.joint_angles_r[0])
        self.knee_l_coordi.update(self.knee_l, self.joint_angles_l[1]+self.joint_angles_l[0])#, self.joint_angles_l[0])
        self.knee_r_coordi.update(self.knee_r, self.joint_angles_r[1]+self.joint_angles_r[0])#, self.joint_angles_r[0])
        self.ankle_l_coordi.update(self.ankle_l, self.joint_angles_l[2]+self.joint_angles_l[1]+self.joint_angles_l[0])#, self.joint_angles_l[1])
        self.ankle_r_coordi.update(self.ankle_r, self.joint_angles_r[2]+self.joint_angles_r[1]+self.joint_angles_r[0])#, self.joint_angles_r[1])

    def forward_kinematics(self):
        l0 = self.link_lengths[0]
        l1 = self.link_lengths[1]
        theta0_l = self.joint_angles_l[0]
        theta1_l = self.joint_angles_l[1]
        self.knee_l = self.hip + np.array([l0*cos(theta0_l), l0*sin(theta0_l)]) # location of a point knee
        self.ankle_l = self.knee_l + np.array([l1*cos(theta0_l + theta1_l), l1*sin(theta0_l + theta1_l)]) # location of a point ankle

        theta0_r = self.joint_angles_r[0]
        theta1_r = self.joint_angles_r[1]
        self.knee_r = self.hip + np.array([l0*cos(theta0_r), l0*sin(theta0_r)]) # location of a point knee
        self.ankle_r = self.knee_r + np.array([l1*cos(theta0_r + theta1_r), l1*sin(theta0_r + theta1_r)]) # location of a point ankle

    def plot(self, style='r-', coordi=True):
        
        plt.plot([self.pelvis[0], self.hip[0]],
                 [self.pelvis[1], self.hip[1]],
                 style)
        plt.plot([self.hip[0], self.knee_l[0]],
                 [self.hip[1], self.knee_l[1]],
                 style)
        plt.plot([self.hip[0], self.knee_r[0]],
                 [self.hip[1], self.knee_r[1]],
                 style+'-')
        plt.plot([self.knee_l[0], self.ankle_l[0]],
                 [self.knee_l[1], self.ankle_l[1]],
                 style)
        plt.plot([self.knee_r[0], self.ankle_r[0]],
                 [self.knee_r[1], self.ankle_r[1]],
                 style+'-')
        plt.plot(self.pelvis[0], self.pelvis[1], 'ko')
        plt.plot(self.hip[0], self.hip[1], 'ko')
        plt.plot(self.knee_l[0], self.knee_l[1], 'ko')
        plt.plot(self.knee_r[0], self.knee_r[1], 'ko')
        plt.plot(self.ankle_l[0], self.ankle_l[1], 'ko')
        plt.plot(self.ankle_r[0], self.ankle_r[1], 'ko')

        if coordi:
            #############
            # plt.arrow(self.pelvis_coordi.origin[0], self.pelvis_coordi.origin[1], self.pelvis_coordi.x_axis[0], self.pelvis_coordi.x_axis[1], color='b')
            # plt.arrow(self.origin[0], self.origin[1], self.y_axis[0], self.y_axis[1], color='b')
            self.pelvis_coordi.plot()
            self.hip_l_coordi.plot()
            self.hip_r_coordi.plot()
            self.knee_l_coordi.plot()
            self.knee_r_coordi.plot()
            self.ankle_l_coordi.plot()
            self.ankle_r_coordi.plot()

        ###################

        
    
    

def _animation():
    fig = plt.figure()
    camera = Camera(fig)
    plt.ion()
    body_exp = TwoLinkArm()
    body_exp.plot()
    body_patho = TwoLinkArm()
    body_patho.plot()
    body_agent = TwoLinkArm()
    body_agent.plot()
    i = 0
    j = 0
    buffer_exp = SerializedBuffer('gail/buffers/sconewalk_origin_motored_normal_h0914-v3/v1/Adjusted_Same70Init_dof_size1000000_for_hamstringweakness_new_freq1.43_100hz.pth', device='cpu')
    buffer_patho = SerializedBuffer('gail/buffers/sconewalk_origin_motored_hamstringweakness_h0914-v3/v1/Same70Init_dof_size1000000_std0.0_prand1.0_100hz.pth', device='cpu')
    # buffer_agent = torch.load('gail/animations/sconewalk_origin_motored_hamstringweakness_new_h0914-v3/2025-04-05_18_34_500.pth')['state'].clone().to('cpu')
    # buffer = SerializedBuffer('gail/buffers/sconewalk_origin_motored_normal_h0914-v3/v1/Same70Init_dof_size1000000_std0.0_prand1.0_100hz.pth', device='cpu')
    states_exp = buffer_exp.states
    states_exp[:, 1] = buffer_exp.obs_tx.squeeze().clone()
    states_patho = buffer_patho.states
    states_patho[:, 1] = buffer_patho.obs_tx.squeeze().clone()
    states_agent = torch.load('gail/animations/sconewalk_origin_motored_hamstringweakness_new_h0914-v3/2025-04-05_19_07_5000000.pth')['state'].clone().to('cpu')
    # pdb.set_trace()
    dofs = [0, 1, 2, 3, 4, 5, 6, 7, 8] # pelvis_tilt, pelvis_tx, pelvis_ty, hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l
    dof_poses_exp = states_exp[:, dofs].cpu().numpy()
    dof_poses_patho = states_patho[:, dofs].cpu().numpy()
    # dofs = [0, 2, 3, 4, 5, 6, 7, 8] # pelvis_tilt, pelvis_tx, pelvis_ty, hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l
    dof_poses_agent = states_agent[:, dofs].cpu().numpy()
    # pdb.set_trace()
    tstart = time.time()
    while True:
        plt.cla()
        i += 1
        pelvis_position_exp = np.array([dof_poses_exp[i, 1], dof_poses_exp[i, 2]])
        joint_angles_l_exp = np.array([dof_poses_exp[i, 3], dof_poses_exp[i, 4], dof_poses_exp[i, 5]])
        joint_angles_r_exp = np.array([dof_poses_exp[i, 6], dof_poses_exp[i, 7], dof_poses_exp[i, 8]])
        body_exp.update_joints(pelvis_position_exp, joint_angles_l_exp, joint_angles_r_exp)
        body_exp.plot('b-')

        pelvis_position_patho = np.array([dof_poses_patho[i, 1], dof_poses_patho[i, 2]])
        joint_angles_l_patho = np.array([dof_poses_patho[i, 3], dof_poses_patho[i, 4], dof_poses_patho[i, 5]])
        joint_angles_r_patho = np.array([dof_poses_patho[i, 6], dof_poses_patho[i, 7], dof_poses_patho[i, 8]])
        body_patho.update_joints(pelvis_position_patho, joint_angles_l_patho, joint_angles_r_patho)
        body_patho.plot('m-')

        pelvis_position_agent = np.array([dof_poses_agent[i, 1], dof_poses_agent[i, 2]])
        joint_angles_l_agent = np.array([dof_poses_agent[i, 3], dof_poses_agent[i, 4], dof_poses_agent[i, 5]])
        joint_angles_r_agent = np.array([dof_poses_agent[i, 6], dof_poses_agent[i, 7], dof_poses_agent[i, 8]])
        body_agent.update_joints(pelvis_position_agent, joint_angles_l_agent, joint_angles_r_agent)
        body_agent.plot('y-')


        plt.xlim(body_exp.pelvis[0]-2, body_exp.pelvis[0]+3)
        plt.ylim(body_exp.pelvis[1]-2, body_exp.pelvis[1]+3)

        # camera.snap()
        plt.show()
        plt.pause(1e-8)
        # print("ankle_y: ", body.ankle_r[1])
        if i == 5000: break
    # animation = camera.animate(fps=60)
    print("Saving")
    # animation.save('animation.mp4')
    print("FPS", 300/(time.time()-tstart))

def test():
    fig = plt.figure()
    camera = Camera(fig)
    for i in range(10):
        plt.plot([i] * 10)
        camera.snap()
    animation = camera.animate()
    animation.save('animation_2.mp4')


if __name__ == "__main__":
    _animation()
    # test()
    
    # animation_plt()
    

"""
Inverse kinematics of a two-joint arm
Left-click the plot to set the goal position of the end effector

Author: Daniel Ingram (daniel-s-ingram)
        Atsushi Sakai (@Atsushi_twi)

Ref: P. I. Corke, "Robotics, Vision & Control", Springer 2017,
 ISBN 978-3-319-54413-7 p102
- [Robotics, Vision and Control]
(https://link.springer.com/book/10.1007/978-3-642-20144-8)

"""

from scipy.spatial.transform import Rotation as Rot


def rot_mat_2d(angle):
    """
    Create 2D rotation matrix from an angle

    Parameters
    ----------
    angle :

    Returns
    -------
    A 2D rotation matrix

    Examples
    --------
    >>> angle_mod(-4.0)


    """
    return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]


def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)

    Parameters
    ----------
    x : float or array_like
        A angle or an array of angles. This array is flattened for
        the calculation. When an angle is provided, a float angle is returned.
    zero_2_2pi : bool, optional
        Change angle modulo range to [0, 2pi)
        Default is False.
    degree : bool, optional
        If True, then the given angles are assumed to be in degrees.
        Default is False.

    Returns
    -------
    ret : float or ndarray
        an angle or an array of modulated angle.

    Examples
    --------
    >>> angle_mod(-4.0)
    2.28318531

    >>> angle_mod([-4.0])
    np.array(2.28318531)

    >>> angle_mod([-150.0, 190.0, 350], degree=True)
    array([-150., -170.,  -10.])

    >>> angle_mod(-60.0, zero_2_2pi=True, degree=True)
    array([300.])

    """
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle


# Simulation parameters
Kp = 15
dt = 0.01

# Link lengths
l1 = l2 = 1

# Set initial goal position to the initial end-effector position
x = 2
y = 0

show_animation = True

if show_animation:
    plt.ion()


def two_joint_arm(GOAL_TH=0.0, theta1=0.0, theta2=0.0):
    """
    Computes the inverse kinematics for a planar 2DOF arm
    When out of bounds, rewrite x and y with last correct values
    """
    global x, y
    x_prev, y_prev = None, None
    while True:
        try:
            if x is not None and y is not None:
                x_prev = x
                y_prev = y
            if np.hypot(x, y) > (l1 + l2):
                theta2_goal = 0
            else:
                theta2_goal = np.arccos(
                    (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2))
            tmp = math.atan2(l2 * np.sin(theta2_goal),
                                (l1 + l2 * np.cos(theta2_goal)))
            theta1_goal = math.atan2(y, x) - tmp

            if theta1_goal < 0:
                theta2_goal = -theta2_goal
                tmp = math.atan2(l2 * np.sin(theta2_goal),
                                    (l1 + l2 * np.cos(theta2_goal)))
                theta1_goal = math.atan2(y, x) - tmp

            theta1 = theta1 + Kp * ang_diff(theta1_goal, theta1) * dt
            theta2 = theta2 + Kp * ang_diff(theta2_goal, theta2) * dt
        except ValueError as e:
            print("Unreachable goal"+e)
        except TypeError:
            x = x_prev
            y = y_prev

        wrist = plot_arm(theta1, theta2, x, y)

        # check goal
        d2goal = None
        if x is not None and y is not None:
            d2goal = np.hypot(wrist[0] - x, wrist[1] - y)

        if abs(d2goal) < GOAL_TH and x is not None:
            return theta1, theta2


def plot_arm(theta1, theta2, target_x, target_y):  # pragma: no cover
    shoulder = np.array([0, 0])
    elbow = shoulder + np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)])
    wrist = elbow + \
        np.array([l2 * np.cos(theta1 + theta2), l2 * np.sin(theta1 + theta2)])

    if show_animation:
        plt.cla()

        plt.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], 'k-')
        plt.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], 'k-')

        plt.plot(shoulder[0], shoulder[1], 'ro')
        plt.plot(elbow[0], elbow[1], 'ro')
        plt.plot(wrist[0], wrist[1], 'ro')

        plt.plot([wrist[0], target_x], [wrist[1], target_y], 'g--')
        plt.plot(target_x, target_y, 'g*')

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

        plt.show()
        plt.pause(dt)

    return wrist


def ang_diff(theta1, theta2):
    # Returns the difference between two angles in the range -pi to +pi
    return angle_mod(theta1 - theta2)


def click(event):  # pragma: no cover
    global x, y
    x = event.xdata
    y = event.ydata


def animation():
    from random import random
    global x, y
    theta1 = theta2 = 0.0
    for i in range(5):
        x = 2.0 * random() - 1.0
        y = 2.0 * random() - 1.0
        theta1, theta2 = two_joint_arm(
            GOAL_TH=0.01, theta1=theta1, theta2=theta2)


def main():  # pragma: no cover
    fig = plt.figure()
    fig.canvas.mpl_connect("button_press_event", click)
    # for stopping simulation with the esc key.
    fig.canvas.mpl_connect('key_release_event', lambda event: [
                           exit(0) if event.key == 'escape' else None])
    two_joint_arm()


if __name__ == "__main__":
    # animation()
    # main()
    pass