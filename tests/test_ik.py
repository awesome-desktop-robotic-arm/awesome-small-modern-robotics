
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_loader import load_robot
from asmr.kinematics import get_forward_kinematics, get_inverse_kinematics
from utils.geometry import make_T, axis_angle_to_mat
from utils.util import timer

@timer
def test_ik_convergence():
    print("\n--- Testing IK Convergence ---")
    
    # Load Robot
    # Fallback logic same as jacobian test
    robot_path = "mujoco_menagerie/franka_fr3/fr3.xml"
    if not os.path.exists(robot_path):
        robot_path = "tests/simple_robot.xml"
    
    print(f"Loading robot from: {robot_path}")
    if not os.path.exists(robot_path):
        print("Robot file not found. Skipping.")
        return

    robot = load_robot(robot_path)

    print(f"Robot loaded: pose at home: {robot.joint_states}")

    # Pick a target link
    link_names = [l.name for l in robot.links]
    target_link = 'fr3_link7' 
    found = any(target_link in name for name in link_names)
    if not found:
        # Try finding the LAST link
        target_link = link_names[-1]
    
    print(f"Target Link: {target_link}")

    # 1. Generate a valid target pose
    n_joints = len(robot.joints)
    q_target = np.random.uniform(-1.0, 1.0, size=n_joints)
    
    # Compute FK for this q
    res = get_forward_kinematics(robot, list(q_target))
    T_target = res[target_link]
    
    print(f"Target q: {q_target}")
    print(f"Target T:\n{T_target}")

    # 2. Initial Guess (perturb target)
    q_init = q_target + np.random.uniform(-0.1, 0.1, size=n_joints)
    
    print(f"Initial guess q: {q_init}")

    # 3. Solve IK
    try:
        q_sol = get_inverse_kinematics(robot, target_link,
            T_target,
            q_init=list(q_init),
            max_iter=100,
            ptol=1e-4,
            rtol=1e-3
        )
    except Exception as e:
        print(f"IK Failed with error: {e}")
        raise e

    print(f"Solution q: {q_sol}")
    
    # 4. Verification
    # Re-compute FK
    res_sol = get_forward_kinematics(robot, list(q_sol))
    T_sol = res_sol[target_link]
    
    # Check Pose Error
    p_err = np.linalg.norm(T_target[:3, 3] - T_sol[:3, 3])
    R_err = T_target[:3, :3] @ T_sol[:3, :3].T
    trace = np.trace(R_err)
    trace = np.clip(trace, -1.0, 3.0) # Numerical stability
    ang_err = np.arccos((trace - 1) / 2)
    
    print(f"Position Error: {p_err}")
    print(f"Orientation Error: {ang_err}")
    
    if p_err < 1e-3 and ang_err < 1e-2:
        print("SUCCESS: IK Converged to target pose.")
        
    else:
        print("FAILURE: IK did not converge.")
        # Note: DLS might get stuck in local minima, but with small perturbation it implies bug if fails.
        raise AssertionError("IK Convergence failed")

if __name__ == "__main__":
    try:
        test_ik_convergence()
        print("\nAll IK tests passed.")
    except Exception as e:
        print(f"\nTest FAILED: {e}")
        sys.exit(1)
