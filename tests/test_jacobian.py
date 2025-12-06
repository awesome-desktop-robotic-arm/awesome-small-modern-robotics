
import os
import sys
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_loader import load_robot
from asmr.kinematics import Kinematics
from utils.robot_class import Robot

def numerical_jacobian(kin: Kinematics, link_name: str, q: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Compute Jacobian numerically using central difference
    """
    n = len(q)
    J_num = np.zeros((6, n))
    
    # Base pose
    # We need to extract position and orientation (as rotation vector or similar for diff)
    # Typically for numerical J:
    # J_v * dq = dx
    # J_w * dq = d(theta) * axis
    
    # We can just use the FK output matrices.
    # T(q)
    
    # For each joint
    for i in range(n):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[i] += epsilon
        q_minus[i] -= epsilon
        
        fk_plus = kin.get_forward_kinematics(q_plus)
        fk_minus = kin.get_forward_kinematics(q_minus)
        
        T_plus = fk_plus[link_name]
        T_minus = fk_minus[link_name]
        
        # Linear velocity: (p_plus - p_minus) / (2 * eps)
        p_plus = T_plus[:3, 3]
        p_minus = T_minus[:3, 3]
        v_i = (p_plus - p_minus) / (2 * epsilon)
        
        # Angular velocity
        # R_plus = R_curr @ R_delta
        # R_delta = R_cur.T @ R_plus
        # axis * angle = rotation_vector(R_delta)
        # w_i = rotation_vector / (2 * eps) ?
        # Be careful with frame. FK returns T_world.
        # w should be in world frame.
        
        # Easier: R_plus * R_minus.T approx I + [w * 2eps]x 
        # R_diff = T_plus[:3, :3] @ T_minus[:3, :3].T
        # w_i approx derived from skew symmetric part of R_diff
        
        # Using rotation vector from scipy would be easier but let's do manual for small angles
        # R_diff = I + [delta_theta]_x
        # delta_theta (vector) = w_i * (2 * epsilon)
        
        # R_diff between q+ and q-
        R_plus = T_plus[:3, :3]
        R_minus = T_minus[:3, :3]
        
        R_diff = R_plus @ R_minus.T 
        # Extract vector from skew-symmetric part
        # [v]_x = (R - R.T) / 2
        S = (R_diff - R_diff.T) / 2.0
        # v = [S[2,1], S[0,2], S[1,0]]
        w_delta_world = np.array([S[2, 1], S[0, 2], S[1, 0]]) 
        # Wait, this R_diff is R_plus * R_minus^T approx R(w*dt) * R_minus * R_minus^T = R(w*dt).
        # Yes, w is in world frame if we do R_plus @ R_minus.T?
        # Let R(t). R(t+dt) = (I + [w]_x dt) R(t)
        # R_plus = (I + [w*2eps]_x) R_minus
        # R_plus @ R_minus.T = I + [w*2eps]_x
        # Correct.
        
        w_i = w_delta_world / (2 * epsilon)
        
        J_num[:3, i] = v_i
        J_num[3:, i] = w_i
        
    return J_num

def test_jacobian_numerical():
    # Try finding FR3 or fallback to simple_robot
    robot_paths = [
        "mujoco_menagerie/franka_fr3/fr3.xml",
        os.path.join(os.path.dirname(__file__), "simple_robot.xml")
    ]
    
    robot_path = None
    for p in robot_paths:
        if os.path.exists(p):
            robot_path = p
            break
            
    if not robot_path:
        print("No robot file found. Skipping test.")
        return
        
    print(f"Loading robot from: {robot_path}")
    robot = load_robot(robot_path)
    kin = Kinematics(robot)
    
    # Test configuration - random or fixed based on num joints
    n_joints = len(robot.joints)
    # q = np.array([0, 0, 0, -1.57, 0, 1.57, 0.785]) # For 7DOF
    np.random.seed(42)
    q = np.random.uniform(-1.0, 1.0, size=n_joints)
    
    print(f"Testing with q: {q}")
    
    # Target link: end-effector or last link
    # Let's pick 'fr3_link7' or 'fr3_hand' if available
    link_names = [l.name for l in robot.links]
    target_link = 'fr3_link6' # usually in fr3 xml
    # Check if exists, else pick last
    found = any(target_link in name for name in link_names)
    if not found:
        target_link = link_names[-1]
    
    print(f"Testing Jacobian for link: {target_link}")
    
    import time
    
    start = time.perf_counter()
    J_analytic = kin.get_jacobian(target_link, list(q))
    dt_analytic = time.perf_counter() - start
    
    start = time.perf_counter()
    J_num = numerical_jacobian(kin, target_link, q)
    dt_num = time.perf_counter() - start
    
    print("\nAnalytic Jacobian:\n", J_analytic)
    print("\nNumerical Jacobian:\n", J_num)
    
    diff = np.abs(J_analytic - J_num)
    print("\nMax Diff:", np.max(diff))
    
    print(f"\nPerformance:")
    print(f"Analytic Time: {dt_analytic*1000:.4f} ms")
    print(f"Numerical Time: {dt_num*1000:.4f} ms")
    print(f"Speedup: {dt_num/dt_analytic:.2f}x")
    
    # Tolerances: 1e-4 for position, maybe looser for orientation numerical approx
    np.testing.assert_allclose(J_analytic, J_num, atol=1e-4, rtol=1e-3)
    print("Optimization passed: Geometric Jacobian matches Numerical differentiation.")

def test_jacobian_default_args():
    robot_paths = [
        "mujoco_menagerie/franka_fr3/fr3.xml",
        os.path.join(os.path.dirname(__file__), "simple_robot.xml")
    ]
    robot_path = next((p for p in robot_paths if os.path.exists(p)), None)
    
    if not robot_path:
         print("Robot file not found. Skipping test.")
         return

    robot = load_robot(robot_path)
    kin = Kinematics(robot)
    
    q = [0.1] * len(robot.joints)
    robot.joint_states = q
    
    target_link = robot.links[-1].name
    J = kin.get_jacobian(target_link)
    assert J.shape == (6, len(robot.joints))
    print("Default args test passed.")

if __name__ == "__main__":
    test_jacobian_numerical()
    test_jacobian_default_args()
