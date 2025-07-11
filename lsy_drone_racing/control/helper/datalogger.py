import os
import csv
from datetime import datetime
import numpy as np

class DataLogger:
    def __init__(self, log_dir="logs"):
        """Initializes the logger, creating a unique directory for the current run."""
        # Create a timestamped directory for the current run
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(log_dir, f"run_{current_time}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Define file paths
        self.state_log_file = os.path.join(self.run_dir, "state_log.csv")
        self.control_log_file = os.path.join(self.run_dir, "control_log.csv")
        self.gates_log_file = os.path.join(self.run_dir, "final_gates.csv")
        self.obstacles_log_file = os.path.join(self.run_dir, "final_obstacles.csv")

        # Prepare the log files with headers
        self._prepare_state_log()
        self._prepare_control_log()

    def _prepare_state_log(self):
        """Opens the state log CSV and writes the header row."""
        self.state_file_handle = open(self.state_log_file, 'w', newline='')
        self.state_writer = csv.writer(self.state_file_handle)
        header = [
            "time", "px", "py", "pz", "vx", "vy", "vz", 
            "roll", "pitch", "yaw", "f_collective", "f_collective_cmd", 
            "r_cmd", "p_cmd", "y_cmd", "theta", "v_theta",
            "ref_x", "ref_y", "ref_z"       # ← neu
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

    def log_state(
        self,
        time: float,
        state_vector: np.ndarray,
        control_vector: np.ndarray = None,
        ref_point: np.ndarray = None,     # ← neu
    ):
        """Logs a single row of state data and optionally control data and ref-point."""
        # State-Daten
        log_data = [f"{time:.4f}"] + [f"{val:.6f}" for val in state_vector]
        
        # Falls Referenzpunkt übergeben, hinten anhängen
        if ref_point is not None:
            log_data += [f"{ref_point[0]:.6f}", f"{ref_point[1]:.6f}", f"{ref_point[2]:.6f}"]
        
        self.state_writer.writerow(log_data)
        
        # Control-Daten wie bisher
        if control_vector is not None:
            control_data = [f"{time:.4f}"] + [f"{val:.6f}" for val in control_vector]
            self.control_writer.writerow(control_data)


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

    def close(self):
        """Closes any open file handles."""
        if hasattr(self, 'state_file_handle') and self.state_file_handle:
            self.state_file_handle.close()
        if hasattr(self, 'control_file_handle') and self.control_file_handle:
            self.control_file_handle.close()