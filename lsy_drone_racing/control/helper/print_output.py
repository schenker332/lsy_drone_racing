"""Helper module for printing simulation information.

This module contains utilities to format and print simulation data
for debugging and monitoring purposes.
"""

import numpy as np

def print_output(obs:dict, tick , freq):
    """Print current observation data for debugging purposes.
    
    Args:
        obs: Dictionary containing the current observation data
        tick: Current simulation tick counter
        freq: Simulation frequency in Hz
    """
    if tick % int(freq * 0.5) == 0:  # Every 0.5 seconds
        print("\n===== SENSOR INPUT (obs) =====")
        pos = obs.get("pos", None)
        quat = obs.get("quat", None)
        vel = obs.get("vel", None)
        ang_vel = obs.get("ang_vel", None)
        target_gate = obs.get("target_gate", None)
        gates_pos = obs.get("gates_pos", None)
        gates_visited = obs.get("gates_visited", None)
        obstacles_pos = obs.get("obstacles_pos", None)
        obstacles_visited = obs.get("obstacles_visited", None)

        if pos is not None:
            print(f"Position         : {np.round(pos, 3)}")
        if quat is not None:
            print(f"Orientation (quat): {np.round(quat, 3)}")
        if vel is not None:
            print(f"Velocity         : {np.round(vel, 3)}")
        if ang_vel is not None:
            print(f"Angular Velocity : {np.round(ang_vel, 3)}")
        if target_gate is not None:
            print(f"Target Gate      : {int(target_gate)}")
        if gates_pos is not None:
            print("Gates Positions  :")
            for i, g in enumerate(gates_pos):
                print(f"  Gate {i:2}: {np.round(g, 3)}")
        if gates_visited is not None:
            print(f"Gates Visited    : {gates_visited}")
        if obstacles_pos is not None:
            print("Obstacles Pos    :")
            for i, o in enumerate(obstacles_pos):
                print(f"  Obs {i:2}: {np.round(o, 3)}")
        if obstacles_visited is not None:
            print(f"Obstacles Hit    : {obstacles_visited}")
        print("==============================\n")