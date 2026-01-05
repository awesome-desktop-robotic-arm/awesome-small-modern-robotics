import mujoco
from pathlib import Path
import numpy as np
from utils.model_loader import load_robot
from utils.geometry import quat_to_mat

def compare_models():
    robot_path = Path("mujoco_menagerie") / "franka_fr3" / "fr3.xml"
    
    # 1. Load ASMR Robot
    robot = load_robot(str(robot_path))
    
    # 2. Load MuJoCo Model
    m = mujoco.MjModel.from_xml_path(str(robot_path))
    d = mujoco.MjData(m) # Need data for FK
    
    print("Running Kinematics Comparison (Home Pose)...")
    if robot.q_home is not None:
        q = robot.q_home
    else:
        q = np.zeros(len(robot.joints))
    
    # ASMR FK
    # from asmr.kinematics import get_forward_kinematics_all_links (removed)
    
    # Let's perform a manual FK pass here to be sure, or inspect `robot` structure
    # Actually, let's use a helper
    
    link_transforms = {}
    
    def compute_global_transforms(link, T_parent):
        T_curr = T_parent @ link.T_origin
        
        # Apply joint? NO. joint is internal to link? 
        # In ASMR/URDF/MJCF, Link Origin is defined by Parent->Child transform.
        # Then Joint rotates the geometry/inertial/children WITHIN T_curr?
        # NO. MJCF: <body> <joint> <body>.
        # Child body is relative to PARENT body.
        # Joint transforms the connection between Parent and Chiild?
        # Usually: T = T_rigid @ T_joint(q).
        
        # Let's check how ASMR assumes the tree works.
        # robot_class.Link: T_origin.
        # dynamics.py: T_parent_child = link.T_origin_inv.
        # It assumes T_origin is FIXED offset.
        # Then it applies joint rotation.
        
        # If I want global pose of the "Body Frame" (where inertial is defined):
        # We need to accumulate transforms.
        # But we must apply Joint Angles along the way.
        
        # Correct Logic:
        # T_global_child = T_global_parent @ T_static_offset @ T_joint_rotation
        pass

    # Basic recursion for verification
    # Using 'q' correctly mapped
    
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    
    print(f"{'Link':<20} | {'Pos (ASMR/MJ)':<30} | {'Dist':<10} | {'Quat Dist':<10}") 
    print("-" * 90)
    
    # Recursive FK
    def recursive_check(link, T_parent):
        # 1. Static Transform from Parent
        T_static = link.T_origin
        
        # 2. Joint Transform
        # Link has joints?
        T_joint_tot = np.eye(4)
        for joint in link.joints:
             q_val = q[robot.joints.index(joint)]
             if joint.type == 'hinge':
                 from utils.geometry import axis_angle_to_mat
                 R_j = axis_angle_to_mat(joint.axis, q_val)
                 T_j = np.eye(4)
                 T_j[:3, :3] = R_j
                 # Joint pos offset? 
                 # In MJCF, joint is in Body frame. Rotation happens IN body frame.
                 # The children are defined relative to Body Frame.
                 # So does moving the joint affect children?
                 # No, unless child is attached 'after' joint.
                 # In MJCF, the body frame ITSELF moves relative to parent.
                 # Re-reading MJCF: "The body frame is the frame of reference for the body's shapes, inertial properties... and child bodies."
                 # "A joint creates a motion degree of freedom between the body and its parent."
                 # So T_body_parent = T_static_offset @ T_joint(q).
                 T_joint_tot = T_joint_tot @ T_j # If multiple joints? (Complicated, usually 1)
        
        # Total Transform of this Link's Body Frame
        T_curr = T_parent @ T_static @ T_joint_tot
        
        # Compare with MuJoCo
        # MJCF body names match link names
        try:
             bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link.name)
             if bid != -1:
                 mj_pos = d.xpos[bid]
                 mj_quat = d.xquat[bid] # w x y z
                 
                 # Calc distances
                 asmr_pos = T_curr[:3, 3]
                 dist = np.linalg.norm(asmr_pos - mj_pos)
                 
                 # Quat dist
                 from utils.geometry import mat_to_axis_angle
                 # (Just check dot product of orientation columns?)
                 # Or check R_asmr vs R_mj
                 R_asmr = T_curr[:3, :3]
                 R_mj = quat_to_mat(mj_quat)
                 rot_diff = np.linalg.norm(R_asmr - R_mj)
                 
                 print(f"{link.name:<20} | {str(np.round(asmr_pos,3))}/{str(np.round(mj_pos,3)):<15} | {dist:.4f}     | {rot_diff:.4f}")
        except:
             pass

        for child in link.children:
            recursive_check(child, T_curr)

    recursive_check(robot.root, np.eye(4))
    
    print("-" * 80)
    print("Running STATIC Gravity Comparison (Home Pose, Zero Vel/Acc)...")
    
    # Dynamics Comparison
    from asmr.dynamics import get_inverse_dynamics
    
    # Disable MuJoCo Friction/Armature
    m.dof_damping[:] = 0
    m.dof_armature[:] = 0
    m.dof_frictionloss[:] = 0
    
    # Use Home Pose
    if robot.q_home is not None:
        q = robot.q_home
    else:
        q = np.zeros(len(robot.joints))
        
    qd = np.zeros(len(robot.joints))
    qdd = np.zeros(len(robot.joints)) # Static
    
    # ASMR
    tau_asmr = get_inverse_dynamics(robot, q, qd, qdd)
    
    # MuJoCo
    d = mujoco.MjData(m)
    d.qpos[:] = q
    d.qvel[:] = qd
    d.qacc[:] = qdd
    
    # Important: MuJoCo Inverse uses qacc to compute Force = M*qacc + C + G.
    # If qacc=0, Force = C + G. If qvel=0, Force = G.
    mujoco.mj_inverse(m, d)
    tau_mujoco = d.qfrc_inverse.copy()
    
    print(f"{'Joint':<20} | {'ASMR':<15} | {'MuJoCo':<15} | {'Diff':<15}")
    print("-" * 80)
    for i, joint in enumerate(robot.joints):
        t_a = tau_asmr[i]
        t_m = tau_mujoco[i]
        diff = t_a - t_m
        print(f"{joint.name:<20} | {t_a: .5f}        | {t_m: .5f}        | {diff: .5f}")

if __name__ == "__main__":
    compare_models()
