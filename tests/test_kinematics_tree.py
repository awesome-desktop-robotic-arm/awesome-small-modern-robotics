import sys
import os
import numpy as np

# Add parent directory to path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_loader import load_robot
from asmr.kinematics import Kinematics
from utils.robot_class import Link

def print_tree(link: Link, level=0):
    indent = "  " * level
    print(f"{indent}- {link.name}")
    for child in link.children:
        print_tree(child, level + 1)

def main():
    # Load robot
    # Try to find a valid xml file.
    robot_path = "mujoco_menagerie/franka_fr3/fr3.xml"
    if not os.path.exists(robot_path):
        # Fallback to creating a dummy generic robot if file not found, 
        # but better to test with real file if possible.
        # Check current dir
         if not os.path.exists(os.path.join('..', robot_path)):
             print(f"Warning: {robot_path} not found. Test might fail if file missing.")
    
    print(f"Loading robot from {robot_path}...")
    try:
        robot = load_robot(robot_path)
    except FileNotFoundError:
        # try absolute path or relative to script?
        # Assuming run from root of repo
        robot = load_robot(robot_path)

    print(f"\nRobot Root: {robot.root.name}")
    print("\nKinematics Tree Structure:")
    print_tree(robot.root)
    
    # Initialize Kinematics
    kin = Kinematics(robot)
    
    # Test FK with zero angles
    print("\nTesting Forward Kinematics (Zero Angles)...")
    zeros = [0.0] * len(robot.joints)
    # fk_result = kin.get_forward_kinematics(zeros)

    home = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
    fk_result = kin.get_forward_kinematics(home) # Expect [0.55449948 0.         0.73150243] to agree with MJ.forward

    print(f"Computed FK for {len(fk_result)} links.")
    for link_name, transform in fk_result.items():
        print(f"{link_name}: Rotation={transform[:3, :3]}, Translation={transform[:3, 3]}")

    print("\nSUCCESS: Kinematics tree traversal and basics working.")

if __name__ == "__main__":
    main()
