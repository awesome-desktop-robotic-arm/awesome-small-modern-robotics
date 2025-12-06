"""
Kinematics module: 
Forward kinematics: walk down kinematics tree
"""

import numpy as np
from typing import Dict, List

from utils.geometry import axis_angle_to_mat, make_T
from utils.robot_class import Robot, Link, Joint

class Kinematics:
    def __init__(self, robot: Robot):
        self.robot = robot
        
    def get_forward_kinematics(self, joint_angles: List[float]) -> Dict[str, np.ndarray]:
        """
        Forward kinematics: walk down kinematics tree
        Args:
            joint_angles (List[float]): List of joint angles matching robot.joints order
        Returns:
            Dict[str, np.ndarray]: Dictionary of link names and their forward kinematics (T_link_world)
        """
        if len(joint_angles) != len(self.robot.joints):
            raise ValueError(f"Number of joint angles ({len(joint_angles)}) must match number of joints ({len(self.robot.joints)})")
    
        result: Dict[str, np.ndarray] = {}
        
        # Start recursion from root
        # We assume root.origin is its transform in world (usually identity for "world" link)
        self._fk_recursive(self.robot.root, self.robot.root.origin, joint_angles, result)
        
        return result

    def _fk_recursive(self, 
        link_current: Link, 
        T_current_world: np.ndarray, 
        joint_angles: List[float],
        result: Dict[str, np.ndarray]):
        """
        Recursively compute forward kinematics

        Args:
            link_current (Link): Current link
            T_current_world (np.ndarray): Transform of current link in world frame
            joint_angles (List[float]): List of joint angles matching robot.joints order
            result (Dict[str, np.ndarray]): Dictionary of link names and their forward kinematics (T_link_world)
        """
        # Store result for current link
        result[link_current.name] = T_current_world
        
        # Iterate over children
        for child in link_current.children:
            
            # 1. Get Static transform from parent (current) to child (before joint motion)
            T_child_static = child.origin
            
            # 2. Compute Joint transforms which happen in the child frame
            T_joint_total = np.eye(4)
            
            # Iterate over all joints that connect parent to this child
            for joint in child.joints:
                idx = self.robot.joints.index(joint)
                q = joint_angles[idx]
                
                # Joint origin relative to child frame
                T_j_origin = joint.origin 
                T_j_origin_inv = np.linalg.inv(T_j_origin)
                
                if joint.type == 'hinge':
                    R_motion = axis_angle_to_mat(joint.axis, q)
                    T_motion = make_T(R_motion, np.zeros(3))
                elif joint.type == 'slide':
                    p_motion = joint.axis * q
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
            self._fk_recursive(child, T_child_world, joint_angles, result)