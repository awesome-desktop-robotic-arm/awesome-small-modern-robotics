"""
Dynamics module for lib ASMR

Forward dynamics: solving second order ODE for equation of motions
Inverse dynamics: recursive Newton-Euler algorithm

"""

import numpy as np
from utils.robot_class import Robot, Link
from utils.util import check_input_dimensions
from utils.geometry import axis_angle_to_mat
from asmr.kinematics import get_jacobian
from typing import Optional, Dict



def forward_dynamics(robot: Robot, q: np.ndarray, qd: np.ndarray, tau: np.ndarray, F_ext: Optional[np.ndarray] = None) -> np.ndarray:
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
    # Check input dimensions
    check_input_dimensions(robot, q, qd, tau)

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
    Compute the inertia matrix of the robot at given joint positions via Column-wise rNEA

    Args:
        robot (Robot): Robot object
        q (np.ndarray): Joint positions (n,)

    Returns:
        np.ndarray: Mass matrix (n, n)
    """
    # Check input dimensions
    check_input_dimensions(robot, q)

    M = np.zeros((len(q), len(q)))
    qd = np.zeros_like(q)
    
    # Compute mass column via ID
    for joint in range(len(q)):
        qdd = np.zeros_like(q)
        qdd[joint] = 1.0
        tau = get_inverse_dynamics(robot=robot, q=q, qd=qd, qdd=qdd, gravity=np.zeros(3)) # Need to turn off gravity to avoid double calc
        M[:, joint] = tau
    
    return M


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
    # Check input dimensions
    check_input_dimensions(robot, q, qd)

    tau = get_inverse_dynamics(robot=robot, q=q, qd=qd, qdd=np.zeros_like(qd))
    tau += get_joint_friction(robot=robot, q=q, qd=qd)
    
    return tau


def get_joint_friction(robot: Robot, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    """
    Compute the joint friction forces of the robot at given joint positions and velocities

    Args:
        robot (Robot): Robot object
        q (np.ndarray): Joint positions (n,)
        qd (np.ndarray): Joint velocities (n,)

    Returns:
        np.ndarray: Joint friction forces (n,)
    """
    # TODO: Implement this later.
    return np.zeros_like(q)


def get_inverse_dynamics(robot: Robot, 
                         q: np.ndarray, 
                         qd: np.ndarray, 
                         qdd: np.ndarray, 
                         external_forces: Optional[Dict[str, np.ndarray]] = None, # {link_name: F_ext}
                         gravity: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute the inverse dynamics of the robot at given joint positions, velocities, and accelerations through recursive Newton-Euler algorithm
    
    Refactored to use Recursion to guarantee topological order (Parents computed before Children).

    Args:
        robot (Robot): Robot object
        q (np.ndarray): Joint positions (n,)
        qd (np.ndarray): Joint velocities (n,)
        qdd (np.ndarray): Joint accelerations (n,)
        external_forces (Optional[Dict[str, np.ndarray]], optional): External forces at each link (6,). Defaults to None.
        gravity (np.ndarray, optional): Gravity vector (3,). Defaults to None to be initialized.

    Returns:
        np.ndarray: Joint torques (n,)
    """
    # Check input dimensions
    check_input_dimensions(robot, q, qd, qdd)

    # Gravity
    if gravity is None:
        gravity = np.array([0, 0, -9.81])
    
    # Initial velocity and acceleration - initial acceleration assumes -gravity
    v_0 = np.zeros(3)
    w_0 = np.zeros(3)
    dv_0 = -gravity
    dw_0 = np.zeros(3)
    
    tau = np.zeros(len(robot.joints))

    def _recursive_rnea(link: Link, v_parent, w_parent, dv_parent, dw_parent):
        """
        Recursive RNEA:
        1. Forward Pass (Pre-order): Compute v, w, dv, dw from parent
        2. Backward Pass (Post-order): Compute f, m from children and project to joints
        Returns: (f_link, m_link) - The total wrench this link exerts on its parent (in Link Frame)
        """
        
        # --- 1. Forward Kinematics (Parent -> Current) ---
        if link == robot.root:
            v_curr, w_curr = v_parent, w_parent
            dv_curr, dw_curr = dv_parent, dw_parent
            
        else:
            # Transform from parent -> current link
            T_parent_child = link.T_origin_inv
            R = T_parent_child[:3, :3]
            p = link.T_origin[:3, 3] # Translation in Parent Frame

            # Propagate velocity through rigid body transform
            v_curr = R @ (v_parent + np.cross(w_parent, p))
            w_curr = R @ w_parent
            
            # Propagate acceleration
            term1 = np.cross(dw_parent, p)
            term2 = np.cross(w_parent, np.cross(w_parent, p))
            dv_curr = R @ (dv_parent + term1 + term2)
            dw_curr = R @ dw_parent

            # Store rotations for backward pass
            joint_rotations = []

            # Apply Joint Motion (Iterate through all joints connecting Parent -> Child)
            for joint in link.joints:
                joint_index = robot.joints.index(joint)
                q_i = q[joint_index]
                qd_i = qd[joint_index]
                qdd_i = qdd[joint_index]

                if joint.type == 'hinge':
                    
                    # Rotate Frame by Joint Angle, need to rotate incoming vectors INTO the new frame.
                    # R_j is local rotation from Joint=0 to Joint=q
                    # v_new = R_j.T @ v_old
                    R_j = axis_angle_to_mat(joint.axis, q_i)
                    joint_rotations.append(R_j)
                    
                    v_curr = R_j.T @ v_curr
                    w_curr = R_j.T @ w_curr
                    dv_curr = R_j.T @ dv_curr
                    dw_curr = R_j.T @ dw_curr

                    z_axis = joint.axis / np.linalg.norm(joint.axis)
                    
                    # Add Joint Velocity
                    w_prev = w_curr.copy() 
                    w_curr += z_axis * qd_i
                    
                    
                    # Update accelerations
                    dw_curr += z_axis * qdd_i + np.cross(w_prev, z_axis * qd_i) # Alpha + Coriolis

                elif joint.type == 'slide':
                    joint_rotations.append(np.eye(3)) # No rotation for slide
                    z_axis = joint.axis / np.linalg.norm(joint.axis)
                    v_curr += z_axis * qd_i
                    # Coriolis: 2 * cross(w, v_rel)
                    dv_curr += z_axis * qdd_i + 2 * np.cross(w_curr, z_axis * qd_i)

        # --- 2. Compute Forces (Inertial + External) ---
        
        # Inertial Forces (F = ma)
        dv_com = dv_curr + np.cross(dw_curr, link.com) + np.cross(w_curr, np.cross(w_curr, link.com))
        
        F_inertial = link.mass * dv_com
        M_inertial = link.inertia @ dw_curr + np.cross(w_curr, link.inertia @ w_curr)
        
        f_total = F_inertial.copy()
        m_total = M_inertial.copy() + np.cross(link.com, F_inertial)
        
        # External Forces
        if external_forces is not None and link.name in external_forces:
            F_ext_user = external_forces[link.name]
            # Assuming F_ext is a LOAD (Force exerted BY robot ON env), we ADD it to required torque.
            f_total += F_ext_user[:3]
            m_total += F_ext_user[3:]

        # --- 3. Recurse to Children (Backward Pass Accumulation) ---
        for child in link.children:
            f_child, m_child = _recursive_rnea(child, v_curr, w_curr, dv_curr, dw_curr)
            
            # Transform Child Wrench -> Current Frame
            T_child_curr = child.T_origin
            R_child_curr = T_child_curr[:3, :3]
            p_child_curr = T_child_curr[:3, 3] # Position of Child in Current
            
            f_child_curr = R_child_curr @ f_child
            m_child_curr = R_child_curr @ m_child + np.cross(p_child_curr, f_child_curr)
            
            f_total += f_child_curr
            m_total += m_child_curr

        # --- 4. Project to Joint Torques AND Un-Rotate ---
        if link != robot.root:
            # Iterate in REVERSE to un-wind rotations
            for i, joint in enumerate(reversed(link.joints)):
                idx = robot.joints.index(joint)
                
                # Note: link.joints is Forward. reversed is Backward.
                # joint_rotations was appended Forward.
                # Corresp rotation index is len - 1 - i
                R_j = joint_rotations[len(joint_rotations) - 1 - i]

                # Shift moment to joint frame (if p_joint != 0)
                p_joint = joint.T_origin[:3, 3] # Usually zero
                m_total -= np.cross(p_joint, f_total)
                
                # Project Torque (Current Frame is aligned with Joint Axis)
                if joint.type == 'hinge':
                    z_axis = joint.axis / np.linalg.norm(joint.axis)
                    tau[idx] = np.dot(m_total, z_axis)
                elif joint.type == 'slide':
                    z_axis = joint.axis / np.linalg.norm(joint.axis)
                    tau[idx] = np.dot(f_total, z_axis)
                
                # Un-Rotate Forces (Back to parent joint / Static frame)
                # f_prev = R_j @ f_curr
                f_total = R_j @ f_total
                m_total = R_j @ m_total

        return f_total, m_total

    # Start overall recursion
    _recursive_rnea(robot.root, v_0, w_0, dv_0, dw_0)
    
    return tau
