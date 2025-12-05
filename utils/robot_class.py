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
    origin: np.ndarray # 4x4 transformation matrix
    parent_link: Optional[str] = None

    def __post_init__(self):
        """post init to check data validity"""
        if self.mass < 0:
            raise ValueError("Mass must be non-negative")
        if self.inertia.shape != (3, 3):
            raise ValueError("Inertia must be a 3x3 matrix")
        if self.origin.shape != (4, 4):
            raise ValueError("Origin must be a 4x4 matrix")
        if self.com.shape != (3,):
            raise ValueError("COM must be a 3-vector")

@dataclass
class Joint:
    name: str
    type: str  # e.g., 'hinge', 'slide' - for mjcf
    parent_link: str
    child_link: str
    axis: np.ndarray  # 3-vector
    origin: np.ndarray  # 4x4 transformation matrix
    limits: Optional[np.ndarray] = None

    def __post_init__(self):
        """post init to check data validity"""
        if self.origin.shape != (4, 4):
            raise ValueError("Origin must be a 4x4 matrix")
        if self.axis.shape != (3,):
            raise ValueError("Axis must be a 3-vector")
        if self.type not in ['hinge', 'slide']:
            raise ValueError("Type must be 'hinge' or 'slide'")

@dataclass
class Robot:
    name: str
    links: List[Link] 
    joints: List[Joint]
