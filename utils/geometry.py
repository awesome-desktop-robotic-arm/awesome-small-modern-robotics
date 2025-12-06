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
    # Handle non-unit quaternions
    q = np.array(q)
    if np.linalg.norm(q) > 1e-6:
        q = q / np.linalg.norm(q)

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

def euler_to_mat(euler: np.ndarray, convention: str = 'radians') -> np.ndarray:
    """
    Converts RPY Euler angles [roll, pitch, yaw] to a rotation matrix.

    Composition: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    :param euler: A numpy array or list with 3 elements [roll, pitch, yaw]
    :param convention: 'radians' or 'degrees'
    :return: A 3x3 numpy array representing the rotation matrix.
    """
    if convention not in ['radians', 'degrees']:
        raise ValueError("Convention must be 'radians' or 'degrees'")

    euler = np.asarray(euler, dtype=float)

    if convention == 'degrees':
        euler = np.deg2rad(euler)

    r, p, y = euler  # roll, pitch, yaw

    sr, cr = np.sin(r), np.cos(r)
    sp, cp = np.sin(p), np.cos(p)
    sy, cy = np.sin(y), np.cos(y)

    R = np.array([
        [cy*cp,   cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,   sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,     cp*sr,             cp*cr           ]
    ])

    return R


def axis_angle_to_mat(axis: np.ndarray, angle: float, convention: str = 'radians') -> np.ndarray:
    """
    Converts an axis-angle representation to a rotation matrix.

    :param axis: A 3-vector representing the rotation axis (should be a unit vector).
    :param angle: The rotation angle.
    :param convention: 'radians' or 'degrees'
    :return: A 3x3 rotation matrix.
    """

    if convention not in ['radians', 'degrees']:
        raise ValueError("Convention must be 'radians' or 'degrees'")

    if convention == 'degrees':
        angle = np.deg2rad(angle)

    # Normalize axis
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis # Unpack
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    # Rodrigues' rotation formula
    K = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]  
    ])

    R = np.eye(3) + s * K + C * (K @ K)
    return R