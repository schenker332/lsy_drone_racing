"""Visualization module for drone racing simulation.

This module provides functions to visualize drone paths, gates, obstacles, and constraints
in the drone racing simulation environment.
"""

import numpy as np

class SimVisualizer:
    """Visualizer class for drone racing simulation.
    
    This class provides methods to visualize various aspects of a drone racing simulation,
    including drone trajectories, gates, obstacles, and planning horizons.
    """

    def __init__(self):
        """Initialize the visualizer with tracking variables."""
        self.flown_positions = []
        self.last_gates_positions = {}
        self.gate_update_points = []
        self.last_obstacles_positions = {}
        self.obstacle_update_points = []


    def reset_episode(self):
        """Reset tracking variables for a new episode."""
        self.flown_positions = []
        self.last_gates_positions = {}
        self.gate_update_points = []
        self.last_obstacles_positions = {}
        self.obstacle_update_points = []


 

    def update_visualization(self, env, obs, controller):
        """Update data and visualize the simulation.
        
        Args:
            env: The simulation environment
            obs: Current observation
            controller: The drone controller
        """
        from lsy_drone_racing.utils import  draw_gates, draw_point, draw_obstacles, draw_line, visualize_cost_weights
        
        def _filter_duplicate_points(points, eps=1e-9):
            if len(points) < 2:
                return points
            keep = [0]
            for i in range(1, len(points)):
                if np.linalg.norm(points[i] - points[keep[-1]]) > eps:
                    keep.append(i)
            return points[keep]

        # Record current drone position (ground truth state)
        drone_pos = obs["pos"]
        self.flown_positions.append(drone_pos)
        trajectories = controller.get_trajectory()
        vis_data = controller.get_contour_lag_error(drone_pos)


        # Extract data for easier access
        ref_point = vis_data['ref_point']
        t_hat_scaled = vis_data['t_hat_scaled']
        e_c_vis = vis_data['e_c_vis']
        e_l_vis = vis_data['e_l_vis']
                

        # # Draw all trajectories except the last one
        # if trajectories and len(trajectories) > 1:
        #     [draw_line(env, traj, rgba=np.array([0.5, 0.8, 0.5, 0.3]), min_size=1.0, max_size=1.0) for traj in trajectories[:-1]]
        # Draw last trajectory
        if trajectories:
            draw_line(env, trajectories[-1], rgba=np.array([0.0, 1.0, 0.0, 1.0]), min_size=2.5, max_size=2.5)



        # Draw flown path
        if len(self.flown_positions) >= 2:
            fp = np.vstack(self.flown_positions)
            draw_line(env, fp, rgba=np.array([1.0, 0.0, 0.0, 1.0]), min_size=1.5, max_size=1.5)


        # Draw planning horizon
        prediction_horizon_points = None
        full_horizon = controller.get_prediction_horizon()
        prediction_horizon_points = full_horizon[::3]  # Take only every third point
        prediction_horizon_points = _filter_duplicate_points(prediction_horizon_points)
        if prediction_horizon_points is not None and len(prediction_horizon_points) >= 2:
            draw_line(env, prediction_horizon_points, rgba=np.array([0.0, 0.8, 1.0, 1.0]), min_size=2.5, max_size=2.5)

        # --- NEW: Visualize cost weights ---
        visualize_cost_weights(env, controller)

        # Draw gates
        if "gates_pos" in obs and "gates_quat" in obs:
            # Draw all gates with their current positions and orientations
            draw_gates(env, 
                    gates_pos=np.array(obs["gates_pos"]), 
                    gates_quat=np.array(obs["gates_quat"]),
                    half_extents=np.array([0.2, 0.015, 0.2]),  # inner opening 0.2 x 0.2, with 0.015 depth of the gate
                    frame_thickness=0.09,  # Beam width in meters
                    rgba_opening=np.array([0.0, 0.7, 1.0, 0.0]),  # Completely transparent (Alpha=0)
                    rgba_frame=np.array([0.0, 0.7, 1.0, 0.5]))   # Light blue
        
        # Draw obstacles
        if "obstacles_pos" in obs:
            # Draw all obstacles as semi-transparent skin-colored cuboids
            draw_obstacles(env, 
                        obstacles_pos=np.array(obs["obstacles_pos"]),
                        width=0.1,        # Width (x-axis)
                        depth=0.1,        # Depth (y-axis)
                        height=2.0,       # Height of the obstacle (z-axis)
                        position_top=True, # Position is top center
                        rgba=np.array([0.94, 0.78, 0.67, 0.5]))  # Semi-transparent skin color
            




        draw_point(env, ref_point, size=0.02, rgba=np.array([1.0, 1.0, 0.0, 0.8]))  # Referenzpunkt (gelb)
        draw_line(env, np.vstack([ref_point, t_hat_scaled]), rgba=np.array([0.0, 0.0, 1.0, 1.0]), min_size=2.0, max_size=2.0)  # Tangentenvektor (blau)
        draw_line(env, np.vstack([ref_point, drone_pos]), rgba=np.array([1.0, 0.0, 0.0, 1.0]), min_size=2.0, max_size=2.0)  # Fehlervektor e (rot)
        draw_line(env, np.vstack([ref_point, e_c_vis]), rgba=np.array([0.0, 1.0, 0.0, 1.0]), min_size=2.0, max_size=2.0)  # Contour Error (grün)
        draw_line(env, np.vstack([ref_point, e_l_vis]), rgba=np.array([1.0, 0.0, 1.0, 1.0]), min_size=2.0, max_size=2.0)  # Lag Error (magenta)

        

        # Draw waypoints
        waypoint_info = controller.get_waypoints()
        waypoints = waypoint_info['waypoints']
        gate_indices = waypoint_info['gate_indices']

        # Draw normal waypoints (small blue points)
        for i, wp in enumerate(waypoints):
            # Check if this waypoint is a gate waypoint
            is_gate_waypoint = False
            for gate_idx in gate_indices:
                if isinstance(gate_idx, int) and i == gate_idx:
                    is_gate_waypoint = True
                    break
            
            if not is_gate_waypoint:
                draw_point(env, wp, size=0.02, rgba=np.array([0.0, 0.0, 1.0, 0.7]))  # Blue

        # Draw gate waypoints (larger red points) - only for gates with center waypoints
        for gate_idx in gate_indices:
            if isinstance(gate_idx, int) and gate_idx < len(waypoints):
                draw_point(env, waypoints[gate_idx], size=0.03, rgba=np.array([1.0, 0.0, 0.0, 0.9]))  # Red

        # Draw orthogonal waypoints (bright red points)
        orthogonal_indices = waypoint_info.get('orthogonal_indices', {})
        for gate_idx, indices in orthogonal_indices.items():
            for direction, wp_idx in indices.items():
                if wp_idx < len(waypoints):
                    draw_point(env, waypoints[wp_idx], size=0.025, rgba=np.array([1.0, 0.0, 0.0, 0.9]))  # Bright red




