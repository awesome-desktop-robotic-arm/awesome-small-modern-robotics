"""
Borrowing mujoco dynamics for use in other function calls.
"""

import mujoco
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_inverse_dynamics(qdd: np.ndarray, qd: np.ndarray, q: np.ndarray, model: mujoco.Any, data: mujoco.Any) -> np.ndarray:
    """
    Get the inverse dynamics using mujoco.

    Args:
        qdd: Joint accelerations.
        qd: Joint velocities.
        q: Joint positions.
        model: Mujoco model.
        data: Mujoco data.

    Returns:
        tau: Joint torques.
    """
    data.qpos[:] = q
    data.qvel[:] = qd
    data.qacc[:] = qdd

    mujoco.mj_inverse(model, data)

    return data.qfrc_inverse.tolist()


def get_bias_forces(model: mujoco.Any, data: mujoco.Any) -> np.ndarray:
    """
    Get the bias forces using mujoco.

    Args:
        qd: Joint velocities.
        q: Joint positions.
        model: Mujoco model.
        data: Mujoco data.

    Returns:
        bias_forces: Bias forces.
    """
    original_qacc = data.qacc.copy()
    data.qacc[:] = 0.0
    bias = np.zeros(model.nv) # number of dof
    mujoco.mj_rne(model, data, 0, bias)
    data.qacc[:] = original_qacc

    return bias



if __name__ == "__main__":
    # Example usage
    model = mujoco.MjModel.from_xml_path("path_to_your_model.xml")
    data = mujoco.MjData(model)

    q = np.zeros(model.nq)
    qd = np.zeros(model.nv)
    qdd = np.ones(model.nv)  # Example accelerations

    tau = get_inverse_dynamics(qdd, qd, q, model, data)
    print("Computed joint torques:", tau)