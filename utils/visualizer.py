"""
Visualization utility for the Robot classes.
Uses matplotlib to render simple 3D representations of robots.
"""

import numpy as np
from matplotlib import pyplot as plt
from robot_class import Robot
from model_loader import load_robot
import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class Visualizer:
    """Simple 3D visualizer for Robot objects."""
    def __init__(self, filepath: str):
        self.robot = load_robot(filepath)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def plot(self):
        """Plot the robot kinematic tree in 3D."""
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # 1. Build Tree (Adjacency List)
        link_map = {l.name: l for l in self.robot.links} # Only name str is hashable now
        children_map = {name: [] for name in link_map}
        
        for link in self.robot.links:
            if link.parent and link.parent.name in link_map:
                children_map[link.parent.name].append(link.name)

        # 2. Recursive Draw Function
        def draw_chain(link_name, T_parent_world):
            link = link_map[link_name]
            
            # Global Transform of this link
            # T_world = T_parent_world * T_local
            T_local = link.origin
            T_world = T_parent_world @ T_local
            
            # Draw Link (Line from Parent Origin to Current Origin)
            p_start = T_parent_world[:3, 3]
            p_end = T_world[:3, 3]
            
            # Plot "Bone"
            self.ax.plot(
                [p_start[0], p_end[0]], 
                [p_start[1], p_end[1]], 
                [p_start[2], p_end[2]], 
                'k-', linewidth=2, alpha=0.5
            )
            
            # Plot Coordinate Frame at Joint/Body Origin
            self._plot_frame(T_world, scale=0.1)

            # Recurse
            for child_name in children_map[link_name]:
                draw_chain(child_name, T_world)

        # Start from world
        draw_chain("world", np.eye(4))
        
        # Auto-scale axes roughly
        self._set_axes_equal(self.ax)
        plt.show()

    def _plot_frame(self, T, scale=0.1):
        """Helper to plot RGB axes for a transform."""
        origin = T[:3, 3]
        # X axis (Red)
        x_axis = origin + T[:3, 0] * scale
        logger.debug(f"x_axis: {x_axis}")
        self.ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], 'r-')
        # Y axis (Green)
        y_axis = origin + T[:3, 1] * scale
        logger.debug(f"y_axis: {y_axis}")
        self.ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], 'g-')
        # Z axis (Blue)
        z_axis = origin + T[:3, 2] * scale
        logger.debug(f"z_axis: {z_axis}")
        self.ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], 'b-')

    def _set_axes_equal(self, ax):
        """Hack to set equal aspect ratio for 3D plots in Matplotlib."""
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


if __name__ == "__main__":
    # Simple test
    vis = Visualizer("mujoco_menagerie/franka_fr3/fr3.xml")
    vis.plot()