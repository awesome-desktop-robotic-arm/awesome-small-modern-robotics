import time
from utils.robot_class import Robot
import numpy as np
from typing import Optional

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Time taken: {end_time - start_time} seconds")
        return result
    return wrapper


def check_input_dimensions(robot: Robot, 
                           q: np.ndarray, 
                           qd: Optional[np.ndarray] = None, 
                           qdd: Optional[np.ndarray] = None, 
                           tau: Optional[np.ndarray] = None):
    if len(q) != len(robot.joint_states):
        raise ValueError("Joint positions must match the number of joints in the robot")
    if qd is not None and len(qd) != len(robot.joint_states):
        raise ValueError("Joint velocities must match the number of joints in the robot")
    if qdd is not None and len(qdd) != len(robot.joint_states):
        raise ValueError("Joint accelerations must match the number of joints in the robot")
    if tau is not None and len(tau) != len(robot.joint_states):
        raise ValueError("Joint torques must match the number of joints in the robot")