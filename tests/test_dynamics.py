import numpy as np
from pathlib import Path

from utils.robot_class import Robot
from utils.model_loader import load_robot
from asmr.dynamics import get_inverse_dynamics, get_mass_matrix, get_bias_forces, forward_dynamics
from asmr.kinematics import get_forward_kinematics

# We will implement these validaton tests one by one:
# 1. test_gravity_compensation
# 2. test_mass_matrix_properties
# 3. test_id_fd_consistency


def test_gravity_compensation():
    """
    Test if gravity compensation is reasonable.
    """

    # Or relative to current working directory
    robot_path = Path("mujoco_menagerie") / "franka_fr3" / "fr3.xml"
    robot = load_robot(str(robot_path))
    print(f"Loaded robot for gravity compensation test: {robot.name}")

    q = robot.q_home
    qd = np.zeros(len(robot.joints))
    
    # Compute gravity compensation torques
    tau_g = get_bias_forces(robot, q, qd)
    
    # Apply them → should have zero acceleration
    qdd = forward_dynamics(robot, q, qd, tau_g)
    
    assert np.allclose(qdd, 0, atol=1e-4), \
        f"Gravity comp failed: qdd = {qdd}"
    print("Gravity compensation test passed.")



def test_mass_matrix_properties():
    """
    Test if mass matrix is symmetric and positive definite.
    """

    # Or relative to current working directory
    robot_path = Path("mujoco_menagerie") / "franka_fr3" / "fr3.xml"
    robot = load_robot(str(robot_path))
    print(f"Loaded robot for mass matrix properties test: {robot.name}")

    M = get_mass_matrix(robot=robot, q=robot.q_home)

    assert np.allclose(M, M.T), "Mass matrix is not symmetric."
    print("Mass matrix is symmetric.")

    # Check if mass matrix is positive definite
    eigvals = np.linalg.eigvals(M)
    assert np.all(eigvals > 0), "Mass matrix is not positive definite."
    print("Mass matrix is positive definite.")


def test_id_fd_consistency():
    """
    Test if inverse dynamics and forward dynamics are consistent.
    """

    # Or relative to current working directory
    robot_path = Path("mujoco_menagerie") / "franka_fr3" / "fr3.xml"
    robot = load_robot(str(robot_path))
    print(f"Loaded robot for ID-FD consistency test: {robot.name}")

    # Random state
    q = np.random.uniform(-0.2, 0.2, len(robot.joints))
    qd = np.random.uniform(-0.2, 0.2, len(robot.joints))
    qdd_desired = np.random.uniform(-2, 2, len(robot.joints))
    
    # Round trip
    tau = get_inverse_dynamics(robot, q, qd, qdd_desired)
    qdd_actual = forward_dynamics(robot, q, qd, tau)
    
    assert np.allclose(qdd_actual, qdd_desired, atol=1e-6), \
        f"ID-FD mismatch: {np.max(np.abs(qdd_actual - qdd_desired))}"
    print("Inverse dynamics and forward dynamics are consistent.")


if __name__ == "__main__":
    test_gravity_compensation()
    test_mass_matrix_properties()
    test_id_fd_consistency()