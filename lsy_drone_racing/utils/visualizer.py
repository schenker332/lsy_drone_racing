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

    @staticmethod
    def update_visualization(env, obs, controller, config, all_trajectories, flown_positions, 
                          last_gates_positions, gate_update_points,
                          last_obstacles_positions, obstacle_update_points):
        """Update data and visualize the simulation.
        
        Args:
            env: The simulation environment
            obs: Current observation
            controller: The drone controller
            config: Configuration object
            all_trajectories: List of all planned trajectories
            flown_positions: List of actual drone positions
            last_gates_positions: Dictionary of last known gate positions
            gate_update_points: List of positions where gates were updated
            last_obstacles_positions: Dictionary of last known obstacle positions
            obstacle_update_points: List of positions where obstacles were updated
        """
        from lsy_drone_racing.utils import draw_line, draw_gates, draw_point, draw_obstacles, generate_parallel_lines
        
        # Record current drone position (ground truth state)
        flown_positions.append(obs["pos"])
        
        # Check if gate positions have changed
        if "gates_pos" in obs:
            gates_pos = obs["gates_pos"]
            for gate_idx, gate_pos in enumerate(gates_pos):
                if gate_idx not in last_gates_positions or not np.array_equal(gate_pos, last_gates_positions[gate_idx]):
                    # A gate position has changed or is observed for the first time
                    gate_update_points.append(obs["pos"])
                    last_gates_positions[gate_idx] = gate_pos.copy()  # Store a copy of the position
        
        # Check if obstacle positions have changed
        if "obstacles_pos" in obs:
            obstacles_pos = obs["obstacles_pos"]
            for obs_idx, obs_pos in enumerate(obstacles_pos):
                if obs_idx not in last_obstacles_positions or not np.array_equal(obs_pos, last_obstacles_positions[obs_idx]):
                    # An obstacle was observed for the first time or has changed position
                    obstacle_update_points.append(obs["pos"])
                    last_obstacles_positions[obs_idx] = obs_pos.copy()  # Store a copy of the position
        
        # Update the planning horizon (predicted positions)
        prediction_horizon_points = None
        try:
            # Get the planning horizon and use only every third point to save computational resources
            full_horizon = controller.get_prediction_horizon()
            prediction_horizon_points = full_horizon[::3]  # Take only every third point
        except Exception as e:
            # If the method is not implemented or an error occurs
            pass
            
        # Draw both the planned path and the flown path every frame
        if config.sim.gui:
            # Draw all previous trajectories with faded colors
            for i, traj in enumerate(all_trajectories[:-1]):
                # Light green with decreasing transparency for older trajectories
                alpha = 0.3 + 0.5 * (i / max(1, len(all_trajectories) - 1))
                draw_line(env, traj,
                        rgba=np.array([0.5, 0.8, 0.5, alpha]),  # Light green with alpha
                        min_size=1.5, max_size=1.5)
            
            # Current planned trajectory: strong green, thickness 2
            if all_trajectories:
                draw_line(env, all_trajectories[-1],
                        rgba=np.array([0.0, 1.0, 0.0, 1.0]),  # Strong green
                        min_size=2.0, max_size=2.0)

            # Visualize the planning horizon: cyan/light blue, thicker line
            if prediction_horizon_points is not None and len(prediction_horizon_points) >= 2:
                draw_line(env, prediction_horizon_points,
                        rgba=np.array([0.0, 0.8, 1.0, 1.0]),  # Cyan/light blue
                        min_size=2.5, max_size=2.5)  # Slightly thicker than the path

            # Actual flown path: red, thickness 1.5
            if len(flown_positions) >= 2:
                fp = np.vstack(flown_positions)
                draw_line(env, fp,
                        rgba=np.array([1.0, 0.0, 0.0, 1.0]),
                        min_size=1.5, max_size=1.5)
                        
            # Draw gate update points as light blue spheres
            for update_point in gate_update_points:
                draw_point(env, update_point, 
                        size=0.03,  # Larger sphere for better visibility
                        rgba=np.array([0.0, 0.7, 1.0, 1.0]))  # Light blue
            
            # Draw obstacle update points as skin-colored spheres
            for update_point in obstacle_update_points:
                draw_point(env, update_point, 
                        size=0.03,  # Larger sphere for better visibility
                        rgba=np.array([0.94, 0.78, 0.67, 1.0]))  # Skin color
            
            # Draw gates with position and orientation
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
                

            # # Draw constraint visualization (tube) only around the relevant part of trajectory
            # # This visualization shows safety boundaries around the prediction horizon
            # if all_trajectories and prediction_horizon_points is not None:
            #     # Current position of the drone
            #     current_pos = obs["pos"]
                
            #     # Find the closest point on the trajectory to the current drone position
            #     traj = np.array(all_trajectories[-1])
            #     distances = np.linalg.norm(traj - current_pos, axis=1)
            #     closest_idx = np.argmin(distances)
                
            #     # Extract the part of the trajectory that corresponds to the prediction horizon
            #     # (from the current position to the length of the horizon)
            #     horizon_length = len(prediction_horizon_points)
            #     end_idx = min(closest_idx + horizon_length, len(traj))
            #     horizon_traj = traj[closest_idx:end_idx]
                
            #     # Draw constraints only around this part of the trajectory
            #     if len(horizon_traj) > 1:  # At least 2 points needed
            #         tube_lines = generate_parallel_lines(horizon_traj, radius=0.25, num_lines=20)
            #         for line in tube_lines:
            #             draw_line(env, line, rgba=np.array([0.8, 0.8, 0.0, 0.7]), min_size=1.0, max_size=1.0)