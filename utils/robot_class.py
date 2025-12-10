"""
Model representation for robots
"""

from dataclasses import dataclass
import dataclasses
import numpy as np
from typing import List, Optional

@dataclass
class Link:
    name: str
    mass: float
    com: np.ndarray # Center of Mass (3-vector)
    inertia: np.ndarray # 3x3 inertia matrix
    T_origin: np.ndarray # 4x4 transformation matrix: transform from parent link frame to this link frame
    T_origin_inv: Optional[np.ndarray] = None # Precomputed inverse of T_origin: transform from this link to parent frame
    # Tree structure
    parent: Optional["Link"] = None # String syntax for forward definition
    children: List["Link"] = dataclasses.field(default_factory=list)
    joints: List["Joint"] = dataclasses.field(default_factory=list) # Joints controlling this link (incoming)
    

    def __post_init__(self):
        """post init to check data validity"""
        if self.mass < 0:
            raise ValueError("Mass must be non-negative")
        if self.inertia.shape != (3, 3):
            raise ValueError("Inertia must be a 3x3 matrix")
        if self.T_origin.shape != (4, 4):
            raise ValueError("Origin must be a 4x4 matrix")
        if self.com.shape != (3,):
            raise ValueError("COM must be a 3-vector")
        
        # Precompute inverse of origin
        self.T_origin_inv = np.linalg.inv(self.T_origin)
        
    def __repr__(self):
        return f"Link(name='{self.name}', children={[c.name for c in self.children]})"

@dataclass
class Joint:
    name: str
    type: str  # e.g., 'hinge', 'slide' - for mjcf
    axis: np.ndarray  # 3-vector
    T_origin: np.ndarray  # 4x4 transformation matrix
    T_origin_inv: Optional[np.ndarray] = None
    # Tree structure
    parent: Optional["Link"] = None
    child: Optional["Link"] = None
    limits: Optional[np.ndarray] = None

    def __post_init__(self):
        """post init to check data validity"""
        if self.T_origin.shape != (4, 4):
            raise ValueError("Origin must be a 4x4 matrix")
        if self.axis.shape != (3,):
            raise ValueError("Axis must be a 3-vector")
        if self.type not in ['hinge', 'slide']:
            raise ValueError("Type must be 'hinge' or 'slide'")
        
        # Precompute inverse of origin
        self.T_origin_inv = np.linalg.inv(self.T_origin)

    def __repr__(self):
        return f"Joint(name='{self.name}', type='{self.type}')"

@dataclass
class Robot:
    name: str
    root: Link
    links: List[Link] 
    joints: List[Joint]
    joint_states: Optional[List[float]] = None
    q_home: Optional[List[float]] = None 

    def __post_init__(self):
        """post init to check data validity"""
        if self.root is None:
            raise ValueError("Root must be a Link")
        if self.links is None:
            raise ValueError("Links must be a list of Link")
        if self.joints is None:
            raise ValueError("Joints must be a list of Joint")
        if self.joint_states is not None:
            if len(self.joint_states) != len(self.joints):
                raise ValueError("Joint states length must match number of joints")
        if self.q_home is not None:
            if len(self.q_home) != len(self.joints):
                raise ValueError("Home configuration length must match number of joints")
        
        # Initialize q_joint to home
        self.joint_states = self.q_home # Will be all zeros if no q_home provided

        # Construct lookup hashmaps for quick access
        self.link_map = {link.name: link for link in self.links}
        self.link_map[self.root.name] = self.root
        self.joint_idx_map = {joint.name: idx for idx, joint in enumerate(self.joints)}
        