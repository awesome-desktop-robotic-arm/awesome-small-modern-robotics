"""
Kinematics module: 
Forward kinematics: walk down kinematics tree
Inverse kinematics: damped least squares

All kinematics functions are now stateless: they will explicitly take in the robot object as an argument.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from utils.geometry import axis_angle_to_mat, make_T, mat_to_axis_angle
from utils.robot_class import Robot, Link, Joint
from utils.util import check_input_dimensions
        
def get_forward_kinematics(robot: Robot, q: List[float]) -> Dict[str, np.ndarray]:
    """
    Forward kinematics: walk down kinematics tree
    Args:
        q (List[float]): List of joint angles matching robot.joints order
    Returns:
        Dict[str, np.ndarray]: Dictionary of link names and their forward kinematics (T_link_world)
    """
    # Check input dimensions
    check_input_dimensions(robot, q)

    result: Dict[str, np.ndarray] = {}
    
    # Start recursion from root
    # We assume root.T_origin is its transform in world (usually identity for "world" link)
    _fk_recursive(robot, robot.root, robot.root.T_origin, q, result)
    
    return result

def _fk_recursive(robot: Robot,
    link_current: Link, 
    T_current_world: np.ndarray, 
    q: List[float],
    result: Dict[str, np.ndarray]):
    """
    Recursively compute forward kinematics

    Args:
        link_current (Link): Current link
        T_current_world (np.ndarray): Transform of current link in world frame
        q (List[float]): List of joint angles matching robot.joints order
        result (Dict[str, np.ndarray]): Dictionary of link names and their forward kinematics (T_link_world)
    """
    # Store result for current link
    result[link_current.name] = T_current_world
    
    # Iterate over children
    for child in link_current.children:
        
        # 1. Get Static transform from parent (current) to child (before joint motion)
        T_child_static = child.T_origin
        
        # 2. Compute Joint transforms which happen in the child frame
        T_joint_total = np.eye(4)
        
        # Iterate over all joints that connect parent to this child
        for joint in child.joints:
            idx = robot.joints.index(joint)
            q_i = q[idx]
            
            # Joint origin relative to child frame <- Mostly an MJCF artifact, although T_j_origin is mostly eye(4)
            T_j_origin = joint.T_origin 
            T_j_origin_inv = joint.T_origin_inv
            
            if joint.type == 'hinge':
                R_motion = axis_angle_to_mat(joint.axis, q_i)
                T_motion = make_T(R_motion, np.zeros(3))
            elif joint.type == 'slide':
                p_motion = joint.axis * q_i
                T_motion = make_T(np.eye(3), p_motion)
            else:
                T_motion = np.eye(4)
                
            # Transform is: Move to Joint Frame -> Apply Motion -> Move back
            T_joint_local = T_j_origin @ T_motion @ T_j_origin_inv
            
            # Compose joint transforms (ordering matters: inner to outer or vice versa? 
            # MJCF applies them sequentially. If multiple joints, usually it's J1 @ J2...)
            T_joint_total = T_joint_total @ T_joint_local
        
        # 3. Combine: T_parent_world @ T_static_child @ T_joint_motion
        T_child_world = T_current_world @ T_child_static @ T_joint_total
        
        # Recurse
        _fk_recursive(robot, child, T_child_world, q, result)

def get_jacobian(robot: Robot, link_name: str, q: Optional[List[float]] = None) -> np.ndarray:
    """
    Compute Geometric Jacobian for a specific link at a specific joint configuration.
    Convention:
        - World frame
        - Output: 6xN matrix ([v; w] if complying with twist syntax, but standard is often [v; w] or [w; v]).
        - Standard geometric Jacobian J = [J_linear; J_angular] (6xN).
        - J_linear = z x (p_e - p_i)
        - J_angular = z
    """
    # 1. Resolve joint angles
    if q is not None:
        check_input_dimensions(robot, q)
    else:
        if robot.joint_states is None:
            raise ValueError("Joint angles not provided and no current configuration set in robot.")
        q = robot.joint_states
        
    # 2. Find link and build chain from root
    target_link = robot.link_map.get(link_name)
    if target_link is None:
            raise ValueError(f"Link {link_name} not found in robot.")
    
    chain: List[Link] = []
    curr = target_link
    while curr is not None:
        chain.append(curr)
        curr = curr.parent
    chain.reverse() # [root, ..., target]
    
    # 3. Forward pass to compute joint axes and end-effector position
    # We need to traverse the chain and accumulate transforms
    
    T_cum = robot.root.T_origin.copy() # Start at root origin
    
    # Store joint info: (index, z_axis_world, p_origin_world, type)
    joint_info: List[Tuple[int, np.ndarray, np.ndarray, str]] = []

    # Iterate through chain
    # Note: chain[0] is root. joints are in children connecting parent->child.
    # So we iterate from the first child (chain[1:])
    
    # Special case: Jacobian of Root. 
    if target_link == robot.root:
        return np.zeros((6, len(robot.joints)))

    # Re-verify FK logic for T_cum
    # In FK: T_child = T_parent @ T_link_static @ T_joints
    
    # Current Logic:
    # T_link_world represents the transform of the current link frame (parent of next child).
    # We step into child:
    # T_curr (start of child chain) = T_link_world @ child.origin (static)
    # Then apply joints sequentially.
    
    T_link_world = T_cum # Current link (starts at root)
    
    for i in range(1, len(chain)):
        child = chain[i]
        
        # 1. Static transform form parent to child
        T_static = child.T_origin
        T_curr = T_link_world @ T_static 
        
        # 2. Process joints for this child
        # Joints are applied sequentially in the child's definition
        
        for joint in child.joints:
            idx = robot.joint_idx_map[joint.name]
            q_i = q[idx]
            
            T_j_origin = joint.T_origin
            
            # Transform to Joint Frame (where axis is defined)
            # The joint rotates around Z (or axis) in this frame.
            # T_curr accumulates previous joints in this link group.
            T_j_world = T_curr @ T_j_origin
            
            # Extract Z axis and Position
            z_axis = T_j_world[:3, :3] @ joint.axis
            p_origin = T_j_world[:3, 3]
            
            joint_info.append((idx, z_axis, p_origin, joint.type))
            
            # Apply motion
            if joint.type == 'hinge':
                R_motion = axis_angle_to_mat(joint.axis, q_i)
                T_motion = make_T(R_motion, np.zeros(3))
            elif joint.type == 'slide':
                p_motion = joint.axis * q_i
                T_motion = make_T(np.eye(3), p_motion)
            else:
                T_motion = np.eye(4)
            
            # Update T_curr for next joint or for link end
            # Strategy: 
            # 1. We are at T_j_world (Joint Frame in World).
            # 2. Apply Motion T_motion (in Joint Frame).
            # 3. Move back to the link frame link utilizing joint.origin_inv.
            #
            # effectively: T_next = T_j_world @ T_motion @ joint.T_origin_inv
            
            T_curr = T_j_world @ T_motion @ joint.T_origin_inv # Need to move back to link frame <- Again an MJCF artifact, mostly eye(4)
            
        # Update T_link_world for next iteration
        T_link_world = T_curr
        
    # T_link_world is now the End-Effector transform
    p_e = T_link_world[:3, 3]
    
    # 4. Assemble Jacobian
    J = np.zeros((6, len(robot.joints)))
    
    for idx, z, p, j_type in joint_info:
        if j_type == 'hinge':
            # J_v = z x (p_e - p)
            J_v = np.cross(z, p_e - p)
            J_w = z
        elif j_type == 'slide':
            J_v = z
            J_w = np.zeros(3)
        else:
            J_v = np.zeros(3)
            J_w = np.zeros(3)
            
        J[:3, idx] = J_v
        J[3:, idx] = J_w
        
    return J


def get_inverse_kinematics(robot: Robot,
                        link_name: str,
                        T_target: np.ndarray, #4x4 transform
                        q_init: List[float] = None,
                        max_iter: int = 100,
                        damping: float = 0.1,
                        ptol: float = 1e-5, #m, 0.1mm
                        rtol: float = 1e-4, #rad, ~0.1deg
                        ) -> List[float]:
    """
    Inverse kinematics: damped least squares - LMA

    args:
        link_name: Name of the link to compute IK for.
        T_target: Target transform in world frame.
        q_init: Initial joint configuration. If None, uses current joint configuration.
        max_iter: Maximum number of iterations.
        damping: Damping parameter for LMA.
        ptol: Position tolerance.
        rtol: Orientation tolerance.

    returns:
        q: List of joint angles that achieve the target transform.
    """

    
    # Ensure link_name is in robot
    target_link = robot.link_map.get(link_name)
    if target_link is None:
        raise ValueError(f"Link {link_name} not found in robot.")

    # Check if q_init is valid or provided. If not, use current joint angles.
    if q_init is not None:
        check_input_dimensions(robot, q_init)
    else:
        if robot.joint_states is None:
            raise ValueError("No joint states provided and no current joint configuration found in robot.")
        q_init = robot.joint_states
    
    # Initialize solution q
    q = q_init.copy()

    # Iterate
    for i in range(max_iter):
        # Forward to find current xpos
        T_curr_dict = get_forward_kinematics(robot, q) #This returns a dict of transforms for each link
        T_curr = T_curr_dict[target_link.name]             

        # Compute error
        # Position
        p_err = T_target[:3, 3] - T_curr[:3, 3]
        # Orientation
        R_err = T_target[:3, :3] @ T_curr[:3, :3].T # Both SO3 so works: R_err @ R_curr = R_target
        axis, angle = mat_to_axis_angle(R_err)
        w_err = axis * angle

        # Assemble to 6-vector
        err = np.concatenate((p_err, w_err))

        # Check tolerance
        if np.linalg.norm(err) < ptol and np.linalg.norm(angle) < rtol: # Or should we use axis angle to find exact angle?
            break

        # find jacobian to link. TODO: Should we use site definition instead? MJCF: Attachment sites
        J = get_jacobian(robot, link_name, q)

        # Inner jac product: jac @ jac.T + damping * I
        JJ = J @ J.T + damping * np.eye(6) # 6DOF for full jac

        # Outer solution for dq: jac.T @ 
        dq = J.T @ np.linalg.solve(JJ, err)

        # Update q
        q += dq
    return q
    # TODO: Add an IK with secondary objective? Null state projection?
    
