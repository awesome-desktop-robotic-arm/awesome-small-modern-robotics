"""
Dynamics module for lib ASMR

Forward dynamics: solving second order ODE for equation of motions
Inverse dynamics: recursive Newton-Euler algorithm

"""

import numpy as np
from utils.robot_class import Robot
from asmr.kinematics import get_forward_kinematics, get_jacobian


def forward_dynamics(robot: Robot, q: np.ndarray, qd: np.ndarray, tau: np.ndarray, F_ext: np.ndarray = None) -> np.ndarray:
    """
    Forward dynamics using Featherstone's Articulated Body Algorithm

    Args:
        robot (Robot): Robot object
        q (np.ndarray): Joint positions (n,)
        qd (np.ndarray): Joint velocities (n,)
        tau (np.ndarray): Joint torques (n,)
        F_ext (np.ndarray, optional): Wrench external force at the end-effector (6,).  [Fx, Fy, Fz, Tx, Ty, Tz]
        Defaults to None.

    Returns:
        np.ndarray: Joint accelerations (n,)
    """

    if F_ext is None:
        tau_ext = np.zeros_like(robot.joint_states)
    else:
        tau_ext = get_jacobian(robot, q).T @ F_ext

    M = get_mass_matrix(robot, q)

    H = get_bias_forces(robot, q, qd)

    # Solve
    qdd = np.linalg.solve(M, tau - H + tau_ext)

    return qdd
    

def get_mass_matrix(robot: Robot, q: np.ndarray) -> np.ndarray:
    """
    Compute the inertia matrix of the robot at given joint positions

    Args:
        robot (Robot): Robot object
        q (np.ndarray): Joint positions (n,)

    Returns:
        np.ndarray: Mass matrix (n, n)
    """
    
    pass

def get_bias_forces(robot: Robot, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """
    Compute the bias forces (Coriolis, centrifugal, gravity, and friction) of the robot at given joint positions and velocities

    Args:
        robot (Robot): Robot object
        q (np.ndarray): Joint positions (n,)
        qd (np.ndarray): Joint velocities (n,)

    Returns:
        np.ndarray: Bias forces (n,)
    """
    
    pass


def get_inverse_dynamics(robot: Robot, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, gravity: np.ndarray = np.array([0, 0, -9.81])) -> np.ndarray:
    """
    Compute the inverse dynamics of the robot at given joint positions, velocities, and accelerations through recursive Newton-Euler algorithm

    Args:
        robot (Robot): Robot object
        q (np.ndarray): Joint positions (n,)
        qd (np.ndarray): Joint velocities (n,)
        qdd (np.ndarray): Joint accelerations (n,)

    Returns:
        np.ndarray: Joint torques (n,)
    """
    # Initial velocity and acceleration - initial acceleration assumes -gravity
    v_0 = np.zeros(3)
    w_0 = np.zeros(3)
    dv_0 = -gravity
    dw_0 = np.zeros(3)
    
    link_velocities = {} # {link_name: (v, w)}
    link_accelerations = {} # {link_name: (dv, dw)}
    link_forces = {} # {link_name: (F, M)}

    # =========== Forward Pass ===========
    for link in robot.links:
        parent = link.parent

        if parent is None:  # Root
            v_parent = v_0
            w_parent = w_0
            dv_parent = dv_0
            dw_parent = dw_0

        else:
            v_parent, w_parent = link_velocities[parent.name]
            dv_parent, dw_parent = link_accelerations[parent.name]

    # Transform from parent -> current link
    T_parent_child = link.T_origin_inv
    R = T_parent_child[:3, :3]
    p = link.T_origin[:3, 3] # need translation in parent frame for cross product!

    v_parent_child = R @ (v_parent + np.cross(w_parent, p))
    w_parent_child = R @ w_parent

    # Joint motion
    for joint in link.joints:
        joint_index = robot.joints.index(joint)
        q_i = q[joint_index]
        qd_i = qd[joint_index]
        qdd_i = qdd[joint_index]

        if joint.type == 'hinge':
            z_axis = joint.axis / np.linalg.norm(joint.axis)
            w_parent_child += z_axis * qd_i
            dw_parent_child += z_axis * qdd_i + np.cross(w_parent_child, z_axis * qd_i) # Acceleration + Coriolis
            v_parent_child += np.cross(w_parent_child, joint.T_origin[:3, 3])
            dv_parent_child += np.cross(dw_parent_child, joint.T_origin[:3, 3]) \
                            + np.cross(w_parent_child, np.cross(w_parent_child, joint.T_origin[:3, 3])) # Tangential + Centrepetal
        elif joint.type == 'slide':
            z_axis = joint.axis / np.linalg.norm(joint.axis)
            v_parent_child += z_axis * qd_i
            dv_parent_child += z_axis * qdd_i + np.cross(w_parent_child, z_axis * qd_i)

    # Inertial Forces
    dv_com = dv_parent_child + np.cross(dw_parent_child, link.com) \
            + np.cross(w_parent_child, np.cross(w_parent_child, link.com))
    
    F_inertial = link.mass * dv_com
    M_inertial = link.inertia @ dw_parent_child + np.cross(w_parent_child, link.inertia @ w_parent_child)

    # Store
    link_velocities[link.name] = (v_parent_child, w_parent_child)
    link_accelerations[link.name] = (dv_parent_child, dw_parent_child)
    link_forces[link.name] = (F_inertial, M_inertial)

    # =========== Backward Pass ===========
    f_ext = {}  # Forces from child links
    m_ext = {}  # Moments from child links
    tau = np.zeros(len(robot.joints))

    # Iterate in reverse order
    for link in reversed(robot.links):

        # Inertial forces
        F_i, N_i = link_forces[link.name]

        # forces from children
        f_total = F_i.copy()
        m_total = N_i.copy() + np.cross(link.com, F_i) # Moment about link origin

        for child in link.children:
            # Transform child wrench to current link frame
            T_child_parent = child.T_origin
            R_child_parent = T_child_parent[:3, :3]
            p_child_parent = T_child_parent[:3, 3]

            f_child_parent = R_child_parent @ f_ext[child.name]
            m_child_parent = R_child_parent @ m_ext[child.name] + np.cross(p_child_parent, f_child_parent)

            f_total += f_child_parent
            m_total += m_child_parent

        # Project to joint space
        for joint in link.joints:
            joint_index = robot.joints.index(joint)

            if joint.type == 'hinge':
                z_axis = joint.axis / np.linalg.norm(joint.axis)
                tau_i = z_axis @ m_total
                # Store torque
                tau[joint_index] = tau_i
            elif joint.type == 'slide':
                z_axis = joint.axis / np.linalg.norm(joint.axis)
                tau_i = z_axis @ f_total
                # Store force
                tau[joint_index] = tau_i

        # Store for parent
        f_ext[link.name] = f_total
        m_ext[link.name] = m_total

    return tau # Done