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
        from lsy_drone_racing.utils import  draw_gates, draw_point, draw_obstacles,draw_line
        
        # Record current drone position (ground truth state)
        drone_pos = obs["pos"]
        self.flown_positions.append(drone_pos)
        trajectories = controller.get_trajectory()
        _,min_traj_pos,_ = controller.compute_min_distance_to_trajectory(drone_pos)
        vis_data = controller.get_visualization_data(drone_pos)


        # # Extract data for easier access
        # ref_point = vis_data['ref_point']
        # t_hat_scaled = vis_data['t_hat_scaled']
        # e_c_vis = vis_data['e_c_vis']
        # e_l_vis = vis_data['e_l_vis']
                

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
        if prediction_horizon_points is not None and len(prediction_horizon_points) >= 2:
            draw_line(env, prediction_horizon_points, rgba=np.array([0.0, 0.8, 1.0, 1.0]), min_size=2.5, max_size=2.5)



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
            

        # Check if gate positions have changed
        if "gates_pos" in obs:
            gates_pos = obs["gates_pos"]
            for gate_idx, gate_pos in enumerate(gates_pos):
                if gate_idx not in self.last_gates_positions or not np.array_equal(gate_pos, self.last_gates_positions[gate_idx]):
                    # A gate position has changed or is observed for the first time
                    self.gate_update_points.append(obs["pos"])
                    self.last_gates_positions[gate_idx] = gate_pos.copy()  # Store a copy of the position

        # Draw gate update points 
        for update_point in self.gate_update_points:
            draw_point(env, update_point, size=0.03, rgba=np.array([0.0, 0.7, 1.0, 1.0]))

        
        # Check if obstacle positions have changed
        if "obstacles_pos" in obs:
            obstacles_pos = obs["obstacles_pos"]
            for obs_idx, obs_pos in enumerate(obstacles_pos):
                if obs_idx not in self.last_obstacles_positions or not np.array_equal(obs_pos, self.last_obstacles_positions[obs_idx]):
                    # An obstacle was observed for the first time or has changed position
                    self.obstacle_update_points.append(obs["pos"])
                    self.last_obstacles_positions[obs_idx] = obs_pos.copy()  # Store a copy of the position
        
        # Draw obstacle update points
        for update_point in self.obstacle_update_points:
            draw_point(env, update_point, size=0.03, rgba=np.array([0.94, 0.78, 0.67, 1.0]))


        # draw_point(env, ref_point, size=0.01, rgba=np.array([1.0, 1.0, 0.0, 0.8]))  # Referenzpunkt (gelb)
        # draw_line(env, np.vstack([ref_point, t_hat_scaled]), rgba=np.array([0.0, 0.0, 1.0, 1.0]), min_size=2.0, max_size=2.0)  # Tangentenvektor (blau)
        # draw_line(env, np.vstack([ref_point, drone_pos]), rgba=np.array([1.0, 0.0, 0.0, 1.0]), min_size=2.0, max_size=2.0)  # Fehlervektor e (rot)
        # draw_line(env, np.vstack([ref_point, e_c_vis]), rgba=np.array([0.0, 1.0, 0.0, 1.0]), min_size=2.0, max_size=2.0)  # Contour Error (grün)
        # draw_line(env, np.vstack([ref_point, e_l_vis]), rgba=np.array([1.0, 0.0, 1.0, 1.0]), min_size=2.0, max_size=2.0)  # Lag Error (magenta)


        
        # Draw line and point for minimum trajectory position
        draw_line(env, np.vstack([drone_pos, min_traj_pos]), rgba=np.array([1.0, 0.5, 0.0, 1.0]), min_size=2.0, max_size=2.0)
        draw_point(env, min_traj_pos, size=0.01, rgba=np.array([1.0, 0.5, 0.0, 0.8]))

        

 # Draw waypoints
        waypoint_info = controller.get_waypoints()
        waypoints = waypoint_info['waypoints']
        gate_indices = waypoint_info['gate_indices']
        
        # Draw normal waypoints (small blue points)
        for i, wp in enumerate(waypoints):
            if i not in gate_indices:  # Only draw non-gate waypoints
                draw_point(env, wp, size=0.02, rgba=np.array([0.0, 0.0, 1.0, 0.7]))  # Blue
        
        # Draw gate waypoints (larger red points)
        for gate_idx in gate_indices:
            if gate_idx < len(waypoints):
                draw_point(env, waypoints[gate_idx], size=0.03, rgba=np.array([1.0, 0.0, 0.0, 0.9]))  # Red