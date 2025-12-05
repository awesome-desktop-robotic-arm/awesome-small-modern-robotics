import xml.etree.ElementTree as ET
import numpy as np
import os

from robot_class import Robot, Link, Joint
from geometry import quat_to_mat

def _parse_mjcf(filepath: str) -> Robot:
    """Parse a MuJoCo MJCF XML file and return a Robot instance.
    
    Args:
        filepath (str): Path to the MJCF XML file.
    Returns:
        Robot: Parsed robot model.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    # Find worldbody
    worldbody = root.find('worldbody')
    if worldbody is None:
        raise ValueError("No worldbody found in MJCF file.")
    
    # Robot stub initialization
    links = []
    joints = []
    
        # 2. Recursive Parsing Function
    def parse_body(xml_element: ET.Element, parent_name: str):
        body_name = xml_element.get('name', f"body_{len(links)}")
        
        # --- Parse Joint (The connection to the parent) ---
        # Note: MJCF bodies can have multiple joints (composite joints), 
        # but let's assume single DOF per body for simplicity first.
        joint_elem = xml_element.find('joint') 
        
        if joint_elem is not None:
            j_name = joint_elem.get('name', f"joint_{len(joints)}")
            j_type = joint_elem.get('type', 'hinge') # 'hinge' is MJCF for revolute
            j_axis = _parse_vec(joint_elem.get('axis', '0 0 1'))

            # Origin transform
            pos = _parse_vec(joint_elem.get('pos', '0 0 0'))
            T_origin = np.eye(4)
            T_origin[:3, 3] = pos
            
            # Create your internal Joint object
            new_joint = Joint(
                name=j_name,
                parent_link=parent_name,
                child_link=body_name,
                type=j_type,
                axis=j_axis,
                origin=T_origin
            )
            joints.append(new_joint)
        else:
            # Logic for fixed joint if needed, or just merge bodies
            pass

        # --- Parse Inertial/Geom ---
        # Using explicit <inertial> tag if present
        inertial_elem = xml_element.find('inertial')
        if  inertial_elem is not None:
            mass = float(inertial_elem.get('mass'))
            inertia_data = _parse_mjcf_inertial(inertial_elem)
            new_link = Link(name=body_name, mass=mass, com=inertia_data['com'], inertia=inertia_data['inertia'])
            links.append(new_link)

        # --- Recurse to Children ---
        for child_body in xml_element.findall('body'):
            parse_body(child_body, parent_name=body_name)

    # 3. Start the recursion
    # Worldbody is the "base" (Link 0)
    links.append(Link(name="world", mass=0, com=np.zeros(3), inertia=np.zeros((3,3))))
    
    for child in worldbody.findall('body'):
        parse_body(child, parent_name="world")

    return Robot(name=root.get('model', 'robot'), links=links, joints=joints)


def _parse_vec(string_vals):
    # Helper to turn "1 0 0" into np.array([1, 0, 0])
    return np.fromstring(string_vals, sep=' ')


def _parse_mjcf_inertial(inertial_xml: ET.Element):
    """
    Parses an MJCF <inertial> tag and returns mass, CoM, and the 3x3 Inertia Matrix.
    """
    # 1. Parse raw strings
    mass = float(inertial_xml.attrib['mass'])
    
    pos_str = inertial_xml.attrib.get('pos', '0 0 0')
    com = _parse_vec(pos_str) # Center of Mass (3,)

    diag_str = inertial_xml.attrib.get('diaginertia', '1 1 1') # Default logic needed?
    i_diag_vals = _parse_vec(diag_str)  # Ixx, Iyy, Izz
    
    quat_str = inertial_xml.attrib.get('quat', '1 0 0 0') # w x y z
    quat = _parse_vec(quat_str)

    # 2. Create Diagonal Inertia Matrix
    # Shape (3, 3) with Ixx, Iyy, Izz on diagonal
    I_principal = np.diag(i_diag_vals)

    # 3. Convert Quaternion to Rotation Matrix
    # MJCF uses [w, x, y, z] convention
    R = quat_to_mat(quat)

    # 4. Rotate Inertia into Body Frame: R * I * R.T
    I_body = R @ I_principal @ R.T

    return {
        'mass': mass,
        'com': com,         # The 'pos' attribute
        'inertia': I_body   # The full 3x3 matrix
    }
    
def load_robot(filepath: str) -> Robot:
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    
    if ext == '.xml':
        return _parse_mjcf(filepath)
    elif ext == '.urdf':
        raise NotImplementedError("URDF support is coming soon.")
    else:
        raise ValueError(f"Unknown robot description format: {ext}")


if __name__ == "__main__":
    # Simple test
    robot = load_robot("mujoco_menagerie/arx_l5/arx_l5.xml")
    # robot = load_robot("mujoco_menagerie/franka_fr3/fr3.xml")
    print(f"Loaded robot: {robot.name}")
    print(f"Number of links: {len(robot.links)}")
    print(f"Number of joints: {len(robot.joints)}")


    for link in robot.links:
        print(f"Link: {link.name}, Mass: {link.mass}, Inertia:\n{link.inertia}")

    for joint in robot.joints:
        print(f"Joint: {joint.name}, Type: {joint.type}, Parent: {joint.parent_link}, Child: {joint.child_link}")   