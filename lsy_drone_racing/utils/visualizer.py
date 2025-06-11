"""Visualization module for drone racing simulation.

This module provides functions to visualize drone paths, gates, obstacles, and constraints
in the drone racing simulation environment.
"""

import numpy as np

class SimVisualizer:
    """Visualizer class for drone racing simulation."""

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
        
        # Prüfe, ob sich Gate-Positionen geändert haben
        if "gates_pos" in obs:
            gates_pos = obs["gates_pos"]
            for gate_idx, gate_pos in enumerate(gates_pos):
                if gate_idx not in last_gates_positions or not np.array_equal(gate_pos, last_gates_positions[gate_idx]):
                    # Eine Gate-Position hat sich geändert oder wird zum ersten Mal beobachtet
                    gate_update_points.append(obs["pos"])
                    last_gates_positions[gate_idx] = gate_pos.copy()  # Kopie der Position speichern
        
        # Prüfe, ob sich Obstacle-Positionen geändert haben
        if "obstacles_pos" in obs:
            obstacles_pos = obs["obstacles_pos"]
            for obs_idx, obs_pos in enumerate(obstacles_pos):
                if obs_idx not in last_obstacles_positions or not np.array_equal(obs_pos, last_obstacles_positions[obs_idx]):
                    # Ein Obstacle wurde zum ersten Mal beobachtet oder hat sich geändert
                    obstacle_update_points.append(obs["pos"])
                    last_obstacles_positions[obs_idx] = obs_pos.copy()  # Kopie der Position speichern
        
        # Aktualisiere den Planungshorizont (vorhergesagte Positionen)
        prediction_horizon_points = None
        try:
            # Hole den Planungshorizont und verwende nur jeden dritten Punkt, um Geoms zu sparen
            full_horizon = controller.get_prediction_horizon()
            prediction_horizon_points = full_horizon[::3]  # Nimm nur jeden dritten Punkt
        except Exception as e:
            # Wenn die Methode nicht implementiert ist oder ein Fehler auftritt
            pass
            
        # Draw both the planned path and the flown path every frame
        if config.sim.gui:
            # Zeichne alle vorherigen Trajektorien in abgeschwächten Farben
            for i, traj in enumerate(all_trajectories[:-1]):
                # Hellgrün mit abnehmender Transparenz für ältere Trajektorien
                alpha = 0.3 + 0.5 * (i / max(1, len(all_trajectories) - 1))
                draw_line(env, traj,
                        rgba=np.array([0.5, 0.8, 0.5, alpha]),  # Hellgrün mit Alpha
                        min_size=1.5, max_size=1.5)
            
            # Aktuelle geplante Trajektorie: kräftig grün, Stärke 2
            if all_trajectories:
                draw_line(env, all_trajectories[-1],
                        rgba=np.array([0.0, 1.0, 0.0, 1.0]),  # Kräftiges Grün
                        min_size=2.0, max_size=2.0)

            # Visualisiere den Planungshorizont: cyan/hellblau, dicker Strich
            if prediction_horizon_points is not None and len(prediction_horizon_points) >= 2:
                draw_line(env, prediction_horizon_points,
                        rgba=np.array([0.0, 0.8, 1.0, 1.0]),  # Cyan/Hellblau
                        min_size=2.5, max_size=2.5)  # Etwas dicker als der Pfad

            # tatsächlich geflogener Pfad: rot, Stärke 1.5
            if len(flown_positions) >= 2:
                fp = np.vstack(flown_positions)
                draw_line(env, fp,
                        rgba=np.array([1.0, 0.0, 0.0, 1.0]),
                        min_size=1.5, max_size=1.5)
                        
            # Gate-Update-Punkte als hellblaue Kugeln zeichnen
            for update_point in gate_update_points:
                draw_point(env, update_point, 
                        size=0.03,  # Größere Kugel für bessere Sichtbarkeit
                        rgba=np.array([0.0, 0.7, 1.0, 1.0]))  # Hellblau
            
            # Obstacle-Update-Punkte als hautfarbene Kugeln zeichnen
            for update_point in obstacle_update_points:
                draw_point(env, update_point, 
                        size=0.03,  # Größere Kugel für bessere Sichtbarkeit
                        rgba=np.array([0.94, 0.78, 0.67, 1.0]))  # Hautfarben
            
            # Gates einzeichnen mit Position und Ausrichtung
            if "gates_pos" in obs and "gates_quat" in obs:
                # Zeichne alle Gates mit ihren aktuellen Positionen und Orientierungen
                draw_gates(env, 
                        gates_pos=np.array(obs["gates_pos"]), 
                        gates_quat=np.array(obs["gates_quat"]),
                        half_extents=np.array([0.2, 0.015, 0.2]),  # innenöffung 0.2 x 0.2, mit 0.015 tief des gates
                        frame_thickness=0.09,  # Balkenbreite in Metern
                        rgba_opening=np.array([0.0, 0.7, 1.0, 0.0]),  # Komplett transparent (Alpha=0)
                        rgba_frame=np.array([0.0, 0.7, 1.0, 0.5]))   # Hellblau
            
            # Obstacles einzeichnen
            if "obstacles_pos" in obs:
                # Zeichne alle Obstacles als semi-transparente rote Quader
                draw_obstacles(env, 
                            obstacles_pos=np.array(obs["obstacles_pos"]),
                            width=0.1,        # Breite (x-Achse)
                            depth=0.1,        # Tiefe (y-Achse)
                            height=2.0,       # Höhe des Obstacles (z-Achse)
                            position_top=True, # Position ist oben mittig (top center)
                            rgba=np.array([0.94, 0.78, 0.67, 0.5]))  # Halbtransparente Hautfarbe
                

            # Zeichne Constraints nur um den Teil der Trajektorie, der dem Prädiktionshorizont entspricht
            if all_trajectories and prediction_horizon_points is not None:
                # Aktuelle Position der Drohne
                current_pos = obs["pos"]
                
                # Finde den nächstgelegenen Punkt auf der Trajektorie zur aktuellen Drohnenposition
                traj = np.array(all_trajectories[-1])
                distances = np.linalg.norm(traj - current_pos, axis=1)
                closest_idx = np.argmin(distances)
                
                # Extrahiere den Teil der Trajektorie, der dem Prädiktionshorizont entspricht
                # (von der aktuellen Position bis zur Länge des Horizonts)
                horizon_length = len(prediction_horizon_points)
                end_idx = min(closest_idx + horizon_length, len(traj))
                horizon_traj = traj[closest_idx:end_idx]
                
                # Zeichne nur Constraints um diesen Teil der Trajektorie
                if len(horizon_traj) > 1:  # Mindestens 2 Punkte benötigt
                    tube_lines = generate_parallel_lines(horizon_traj, radius=0.25, num_lines=20)
                    for line in tube_lines:
                        draw_line(env, line, rgba=np.array([0.8, 0.8, 0.0, 0.7]), min_size=1.0, max_size=1.0)