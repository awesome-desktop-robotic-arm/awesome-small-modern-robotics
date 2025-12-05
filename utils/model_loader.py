import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path

from robot_class import Robot, Link, Joint
from geometry import quat_to_mat, make_T


class MJCFParser:
    """Parser for MuJoCo MJCF XML files."""
    def __init__(self):
        # States
        self.links = []
        self.joints = []

    def parse(self, filepath: str) -> Robot:
        """Parse an MJCF file and return a Robot object."""
        tree = ET.parse(filepath)
        root = tree.getroot()
        worldbody = root.find('worldbody')
        
        # handle empty worldbody
        if worldbody is None:
            raise ValueError("No worldbody found in MJCF file.")

        # Add worldbody as a link
        self.links.append(Link(name="world", mass=0, com=np.zeros(3), inertia=np.zeros((3,3)), origin=np.eye(4)))
        
        # Recursive call to parse children
        for child in worldbody.findall('body'):
            self._parse_body(child, parent_name="world")

        # Construct robot
        robot = Robot(name=root.get('model', 'robot'), links=self.links, joints=self.joints)

        return robot
        
    def _parse_body(self, xml_element: ET.Element, parent_name: str):
        """Recursively parse a <body> element and its children."""
        body_name = xml_element.get('name', f"body_{len(self.links)}")
        
        # --- 1. Parse Body Transform (Link Origin) ---
        # "pos" and "quat" on <body> define the frame relative to parent
        body_pos = self._parse_vec(xml_element.get('pos', '0 0 0'))
        body_quat = self._parse_vec(xml_element.get('quat', '1 0 0 0'))
        
        T_link = make_T(quat_to_mat(body_quat), body_pos)

        # --- 2. Parse Inertial Properties ---
        inertial_elem = xml_element.find('inertial')
        if inertial_elem is not None:
            # Parse explicitly
            inertia_data = self._parse_mjcf_inertial(inertial_elem)
            mass = inertia_data['mass']
            com = inertia_data['com']
            inertia_mat = inertia_data['inertia']
        else:
            # Default / Dummy values for bodies without mass (e.g. worldbody or dummy frames)
            mass = 0.0
            com = np.zeros(3)
            inertia_mat = np.zeros((3, 3))
        # TODO: Handle deprecated <fullinertia> tag if needed. Also consider diagonal inertia tags.

        # --- 3. Create Link ---
        new_link = Link(
            name=body_name, 
            mass=mass, 
            com=com, 
            inertia=inertia_mat,
            origin=T_link
        )
        self.links.append(new_link)

        # --- 4. Parse Joint (Connection to Parent) ---
        # In MJCF, joint is defined INSIDE the child body, 
        # specifying the DOF between this body and its parent.
        # TODO: Handle multiple joints per body if needed.
        joint_elem = xml_element.find('joint') 
        if joint_elem is not None:
            j_name = joint_elem.get('name', f"joint_{len(self.joints)}")
            j_type = joint_elem.get('type', 'hinge') 
            j_axis = self._parse_vec(joint_elem.get('axis', '0 0 1'))

            # Joint position is relative to the BODY frame
            j_pos = self._parse_vec(joint_elem.get('pos', '0 0 0'))
            
            T_joint_origin = np.eye(4)
            T_joint_origin[:3, 3] = j_pos
            
            new_joint = Joint(
                name=j_name,
                parent_link=parent_name,
                child_link=body_name,
                type=j_type,
                axis=j_axis,
                origin=T_joint_origin
            )
            self.joints.append(new_joint)

        # --- 5. Recurse to Children ---
        for child_body in xml_element.findall('body'):
            self._parse_body(child_body, parent_name=body_name)

    def _parse_vec(self, string_vals):
        """Helper to turn "1 0 0" into np.array([1, 0, 0])"""
        return np.fromstring(string_vals, sep=' ')


    def _parse_mjcf_inertial(self, inertial_xml: ET.Element):
        """
        Parses an MJCF <inertial> tag and returns mass, CoM, and the 3x3 Inertia Matrix.
        """
        # 1. Parse raw strings
        mass = float(inertial_xml.attrib['mass'])
        
        pos_str = inertial_xml.attrib.get('pos', '0 0 0')
        com = self._parse_vec(pos_str) # Center of Mass (3,)

        diag_str = inertial_xml.attrib.get('diaginertia', '1 1 1') # Default logic needed?
        i_diag_vals = self._parse_vec(diag_str)  # Ixx, Iyy, Izz
        
        quat_str = inertial_xml.attrib.get('quat', '1 0 0 0') # w x y z
        quat = self._parse_vec(quat_str)

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
    """Factory function to load a robot from a file."""
    ext = Path(filepath).suffix
    ext = ext.lower()
    
    if ext == '.xml':
        # Assume MJCF for .xml files
        # Instantiate MJCF Parser
        parser = MJCFParser()
        return parser.parse(filepath)
    elif ext == '.urdf':
        # TODO: Implement URDF parser
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