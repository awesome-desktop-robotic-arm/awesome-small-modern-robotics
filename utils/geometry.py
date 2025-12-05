"""
Geometrical utility functions.
"""

import numpy as np



def quat_to_mat(q: list) -> np.ndarray:
    """
    Converts unit quaternion to rotation matrix.
    
    :param q: A numpy array or list with 4 elements representing a unit quaternion [w, x, y, z] (mjcf convention)
    :return: A 3x3 numpy array representing the rotation matrix.
    """


    w, x, y, z = q[0], q[1], q[2], q[3]

    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])
    
    return R


def make_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Assembling R and p into a homogeneous transformation matrix T.

    :param R: A 3x3 rotation matrix. SO(3)
    :param p: A 3-vector representing translation.
    :return: A 4x4 homogeneous transformation matrix. SE(3)
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T