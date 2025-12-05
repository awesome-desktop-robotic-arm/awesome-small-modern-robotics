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

def euler_to_mat(euler: np.ndarray) -> np.ndarray:
    """
    Converts euler angles (xyz convention) to rotation matrix.
    
    :param euler: A numpy array or list with 3 elements [r, p, y]
    :return: A 3x3 numpy array representing the rotation matrix.
    """
    ai, aj, ak = euler[0], euler[1], euler[2]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    R = np.array([
        [cj*ck, cj*sk, -sj],
        [sj*sc - cs, sj*ss + cc, cj*si],
        [sj*cc + ss, sj*cs - sc, cj*ci]
    ])
    return R