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
    # TODO: This needs parent transformation to build kinematics -> also modify model loader

@dataclass
class Joint:
    name: str
    type: str  # e.g., 'revolute', 'prismatic'
    parent_link: str
    child_link: str
    axis: np.ndarray  # 3-vector
    origin: np.ndarray  # 4x4 transformation matrix
    limits: Optional[np.ndarray] = None

@dataclass
class Robot:
    name: str
    links: List[Link] 
    joints: List[Joint]
