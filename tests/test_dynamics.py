import numpy as np
from pathlib import Path

from utils.robot_class import Robot
from utils.model_loader import load_robot
from asmr.dynamics import get_inverse_dynamics, get_mass_matrix, get_bias_forces, forward_dynamics


def test_gravity_compensation():
    """
    Test if gravity compensation is reasonable.
    """
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


def test_pure_inertial_scaling():
    """
    Inertia should scale linearly
    """
    robot_path = Path("mujoco_menagerie") / "franka_fr3" / "fr3.xml"
    robot = load_robot(str(robot_path))
    print(f"Loaded robot for gravity compensation test: {robot.name}")
    q = robot.q_home
    
    # Test linear scaling
    for scale in [0.5, 1.0, 2.0]:
        qdd = np.array([1.0, 0, 0, 0, 0, 0, 0]) * scale
        tau = get_inverse_dynamics(robot, q, np.zeros_like(q), qdd, gravity=np.zeros(3))
        
        # Should scale linearly
        if scale == 1.0:
            tau_ref = tau
        else:
            tau_ref = tau * (1.0 / scale)

        assert np.allclose(tau, tau_ref * scale, rtol=0.01), "Pure inertial scaling failed."


def test_mass_matrix_properties():
    """
    Test if mass matrix is symmetric and positive definite.
    """
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


def benchmark_against_mujoco():
    """
    Benchmark asmr dynamics against MuJoCo's built-in dynamics.
    """
    
    import mujoco
    robot_path = Path("mujoco_menagerie") / "franka_fr3" / "fr3.xml"
    robot_asmr = load_robot(str(robot_path))

    model = mujoco.MjModel.from_xml_path(str(robot_path))
    data = mujoco.MjData(model)

    # --- KEY CHANGE: Disable Friction/Armature/Damping in MuJoCo for RB-Comparison ---
    model.dof_damping[:] = 0
    model.dof_armature[:] = 0
    model.dof_frictionloss[:] = 0
    # ---------------------------------------------------------------------------------

    q = robot_asmr.q_home
    qd = np.zeros(len(robot_asmr.joints))
    qdd = np.random.uniform(-0.5, 0.5, len(robot_asmr.joints))

    # ASMR inverse dynamics
    tau_asmr = get_inverse_dynamics(robot_asmr, q, qd, qdd)
    # MuJoCo inverse dynamics
    data.qpos[:] = q
    data.qvel[:] = qd
    data.qacc[:] = qdd
    mujoco.mj_inverse(model, data)

    # log results
    tau_mujoco = data.qfrc_inverse.copy()   
    
    print(f"MuJoCo computed tau: {tau_mujoco}")
    print(f"ASMR computed tau:   {tau_asmr}")
    print(f"Difference:          {tau_asmr - tau_mujoco}")


    assert np.allclose(tau_asmr, tau_mujoco, atol=1e-5), \
        f"Benchmark failed: max diff = {np.max(np.abs(tau_asmr - tau_mujoco))}"
    print("Benchmark against MuJoCo passed.")

if __name__ == "__main__":
    test_pure_inertial_scaling()
    test_gravity_compensation()
    test_mass_matrix_properties()
    test_id_fd_consistency()
    benchmark_against_mujoco()