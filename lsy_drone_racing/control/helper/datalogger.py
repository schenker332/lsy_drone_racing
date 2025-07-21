import os
import csv
from datetime import datetime
import numpy as np

class DataLogger:
    def __init__(self):
        """Initializes the logger, creating a unique directory for the current run."""


    def _prepare_state_log(self):
        """Opens the state log CSV and writes the header row."""
        self.state_file_handle = open(self.state_log_file, 'w', newline='')
        self.state_writer = csv.writer(self.state_file_handle)
        header = [
            "time", "px", "py", "pz", "vx", "vy", "vz", 
            "roll", "pitch", "yaw", "f_collective", "f_collective_cmd", 
            "r_cmd", "p_cmd", "y_cmd",
            "ref_x", "ref_y", "ref_z",
            "curvature", "v_theta" # NEW: Add curvature to the state log
        ]
        self.state_writer.writerow(header)

    def _prepare_control_log(self):
        """Opens the control log CSV and writes the header row."""
        self.control_file_handle = open(self.control_log_file, 'w', newline='')
        self.control_writer = csv.writer(self.control_file_handle)
        header = [
            "time", "df_cmd", "dr_cmd", "dp_cmd", "dy_cmd", "dv_theta_cmd"
        ]
        self.control_writer.writerow(header)

    def _prepare_weight_log(self):
        """Opens the weight log CSV and writes the header row."""
        self.weight_file_handle = open(self.weight_log_file, 'w', newline='')
        self.weight_writer = csv.writer(self.weight_file_handle)
        header = [
            "time", "current_weight", "min_weight_so_far", "max_weight_so_far",
            "e_contour", "e_lag"  # NEW: Add error metrics
        ]
        self.weight_writer.writerow(header)

    def log_state(
        self,
        time: float,
        state_vector: np.ndarray,
        control_vector: np.ndarray = None,
        ref_point: np.ndarray = None,
        curvature: float = None,  # NEW
        v_theta: float = None  # NEW: Assuming v_theta is a class attribute
    ):
        """Logs a single row of state data and optionally control data and ref-point."""
        # State-Daten
        log_data = [f"{time:.4f}"] + [f"{val:.6f}" for val in state_vector]
        
        # Falls Referenzpunkt übergeben, hinten anhängen
        if ref_point is not None:
            log_data += [f"{ref_point[0]:.6f}", f"{ref_point[1]:.6f}", f"{ref_point[2]:.6f}"]
        else:
            log_data += ["", "", ""]
        
        # Add curvature and distance
        if curvature is not None:
            log_data.append(f"{curvature:.6f}")
        else:
            log_data.append("")

        log_data.append(f"{v_theta:.6f}")  # Assuming v_theta is a class attribute

        self.state_writer.writerow(log_data)
        
        # Control-Daten wie bisher
        if control_vector is not None:
            control_data = [f"{time:.4f}"] + [f"{val:.6f}" for val in control_vector]
            self.control_writer.writerow(control_data)

    def log_weight_data(
        self,
        time: float,
        current_weight: float,
        e_contour: float = None,
        e_lag: float = None
    ):
        """Logs weight and error data."""
        # Update min/max tracking
        self.min_weight = min(self.min_weight, current_weight)
        self.max_weight = max(self.max_weight, current_weight)
        weight_range = self.max_weight - self.min_weight
        
        # Log weight data with statistics as columns
        weight_data = [
            f"{time:.4f}",
            f"{current_weight:.6f}",
            f"{self.min_weight:.6f}",
            f"{self.max_weight:.6f}",
            f"{weight_range:.6f}"
        ]
        
        # Add error metrics if available
        if e_contour is not None:
            weight_data.append(f"{e_contour:.6f}")
        else:
            weight_data.append("")
            
        if e_lag is not None:
            weight_data.append(f"{e_lag:.6f}")
        else:
            weight_data.append("")
        
        self.weight_writer.writerow(weight_data)

    def log_final_positions(self, gates_pos: np.ndarray, obstacles_pos: np.ndarray):
        """Logs the final positions of gates and obstacles to separate CSV files."""
        # Log final gate positions
        if gates_pos is not None:
            with open(self.gates_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["gate_idx", "x", "y", "z"])
                for i, pos in enumerate(gates_pos):
                    writer.writerow([i] + list(pos))
        
        # Log final obstacle positions
        if obstacles_pos is not None:
            with open(self.obstacles_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["obstacle_idx", "x", "y", "z"])
                for i, pos in enumerate(obstacles_pos):
                    writer.writerow([i] + list(pos))

        # # Add final weight statistics as rows at the end of the weight log file
        # if hasattr(self, 'weight_writer') and self.weight_writer:
        #     # Write separator row
        #     self.weight_writer.writerow(["--- FINAL STATISTICS ---", "", "", "", "", "", ""])
            
        #     # Write the final statistics as rows
        #     self.weight_writer.writerow([
        #         "FINAL_MIN_WEIGHT", 
        #         f"{self.min_weight:.6f}", 
        #         "", "", "", "", ""
        #     ])
        #     self.weight_writer.writerow([
        #         "FINAL_MAX_WEIGHT", 
        #         f"{self.max_weight:.6f}", 
        #         "", "", "", "", ""
        #     ])
        #     self.weight_writer.writerow([
        #         "FINAL_WEIGHT_RANGE", 
        #         f"{self.max_weight - self.min_weight:.6f}", 
        #         "", "", "", "", ""
        #     ])

    def close(self):
        """Closes any open file handles."""
        if hasattr(self, 'state_file_handle') and self.state_file_handle:
            self.state_file_handle.close()
        if hasattr(self, 'control_file_handle') and self.control_file_handle:
            self.control_file_handle.close()
        if hasattr(self, 'weight_file_handle') and self.weight_file_handle:
            self.weight_file_handle.close()
