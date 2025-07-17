from __future__ import annotations 
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control import Controller
if TYPE_CHECKING:
    from numpy.typing import NDArray
from lsy_drone_racing.control.create_ocp_solver import create_ocp_solver
from lsy_drone_racing.control.helper.print_output import print_output
from lsy_drone_racing.control.helper.datalogger import DataLogger
from datetime import datetime


class MPController(Controller):
    """Model Predictive Controller using collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the MPC attitude controller.

        Args:
            obs: Initial observation of the environment's state.
            info: Additional environment information from the reset.
            config: Configuration of the environment.
        """
        super().__init__(obs, info, config)
        self.freq = config.env.freq
        self._tick = 0

        # MPC parameters
        self.N = 50                    # Number of discretization steps
        self.T_HORIZON = 1.5           # Time horizon in seconds
        self.dt = self.T_HORIZON / self.N  # Step size

        waypoints = np.array(
        [
            [1.0, 1.5, 0.2],
            [0.8, 1.0, 0.2],
            [0.7, 0.1, 0.4],
            [0.45, -0.5, 0.56],  # gate1
            [0.2, -0.7, 0.65],
            [0.5, -1.5, 0.8],
            [1, -1.05, 1.2],    # gate2
            [1.15, -0.75, 1],
            [0.5, 0, 0.8],
            [0, 1, 0.66],        # gate3
            [-0.2, 1.4, 0.66],
            [-0.9, 1.3, 0.8], 
            [-0.5, 0, 1.11],     # gate4
            [-0.1, -1, 1.2] 

        ])


        
        ts = np.linspace(0, 1, np.shape(waypoints)[0])
        self.cs_x = CubicSpline(ts, waypoints[:, 0])
        self.cs_y = CubicSpline(ts, waypoints[:, 1])
        self.cs_z = CubicSpline(ts, waypoints[:, 2])


        self.theta = 0
        self.v_theta = 1/ (5 * self.dt * self.freq) 

        gate_indices = [3, 6, 9, 11, 12]
        self.gate_thetas = [ts[i] for i in gate_indices]

        self.gate_peak_weights = [35, 50, 60, 5, 120] 


        # Create the optimal control problem solver
        self.acados_ocp_solver, self.ocp = create_ocp_solver(self.T_HORIZON, self.N)

        # Initialize state variables    
        self.last_f_collective = 0.3  # Current thrust value
        self.last_f_cmd = 0.3     # Commanded thrust value
        self.last_rpy_cmd = np.zeros(3)  # Commanded roll, pitch, yaw

        # Store configuration and tracking variables
        self.config = config
        self.finished = False
        self._info = info
        self.waypoints = waypoints
        self._path_log = []
        
        
        ### ========== logger ==========================
        self._log_every = 1
        self._xcurrent_log: list[tuple[float, np.ndarray]] = []   # (t, xcurrent)
        ### ============================================




    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: Current observation of the environment's state.
            info: Optional additional information.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """

        if self.theta >= 1.2:
            self.finished = True
            



        # Construct the current state vector for the MPC solver
        xcurrent = np.concatenate((obs["pos"], obs["vel"],  R.from_quat(obs["quat"]).as_euler("xyz", degrees=False), [self.last_f_collective, self.last_f_cmd], self.last_rpy_cmd, [ self.theta, self.v_theta]) )
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)



        min_dist,_,min_theta = self.compute_min_distance_to_trajectory(obs["pos"])



        # Prepare reference trajectory and weights for all steps in the horizon
        for j in range(self.N + 1):
            theta_j = min(self.theta + j * self.v_theta * self.dt, 1.0)
            theta_j_next = min(theta_j + 0.0001, 1.0)
            theta_min_j = min(min_theta + j * self.v_theta * self.dt, 1.0)


            p_ref = np.array([
            self.cs_x(theta_j), self.cs_y(theta_j), self.cs_z(theta_j), # for contouring
            self.cs_x(theta_j_next), self.cs_y(theta_j_next), self.cs_z(theta_j_next), #for contouring
            self.get_weight(theta_min_j),   # for gauss weighting
            self.cs_x(theta_min_j), self.cs_y(theta_min_j), self.cs_z(theta_min_j) # for real minimun distance
            ])
            self.acados_ocp_solver.set(j, "p", p_ref)


        ### ========== logger ==========================
        curr_sim_time = self._tick / self.freq   # reale Simulationszeit (s)
        self._log_xcurrent(curr_sim_time, xcurrent)
        ### ============================================

        # Solve the MPC problem
        self.acados_ocp_solver.solve()
        x1 = self.acados_ocp_solver.get(1, "x")
        u1 = self.acados_ocp_solver.get(1, "u")

        ### Debugging block for flying commands

        # Debugging of  feedback law i.e print u1
        # input_names = ["df_cmd", "dr_cmd", "dp_cmd", "dy_cmd", "dv_theta_cmd"]
        # for name, value in zip(input_names, u1):
        #     print(f"{name}: {value:.8f}")

        # print("=" * 20)

       
        # print_output(tick=self._tick, obs=obs, freq=self.config.env.freq)
        
        ## Debugging prinzs for state variables
        state_names = ["px", "py", "pz", "vx", "vy", "vz", "roll", "pitch", "yaw",
                       "f_collective", "f_collective_cmd", "r_cmd", "p_cmd", "y_cmd",
                       "theta", "v_theta"]
        
        # for name, value in zip(state_names, x1):
        #     print(f"{name}: {value}")
        
        # # print(f"weight: {self.get_weight(min_theta):.2f}")
        # print(f"min_dist: {min_dist:.2f} at theta: {min_theta:.2f}")
        # print(f"Time: {self._tick/self.freq:.2f}s")
        # print("=" * 20)


        w = 1 / self.config.env.freq / self.dt
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]  

        self.v_theta = x1[15] 
        self.theta = min(1.0, self.theta + self.v_theta * self.dt)
  
  
        cmd = x1[10:14]
        
        return cmd




    def step_callback(self, action: NDArray[np.floating], obs: dict[str, NDArray[np.floating]], 
                    reward: float, terminated: bool, truncated: bool, info: dict) -> bool:
        """Callback after each environment step.
        
        Args:
            action: The action that was taken.
            obs: Current observation after the action.
            reward: Reward received.
            terminated: Whether the episode terminated.
            truncated: Whether the episode was truncated.
            info: Additional information.
            
        Returns:
            Whether the episode is finished.
        """
        self._tick += 1
        self._info = obs
        # Store drone position for trajectory logging
        self._path_log.append(obs["pos"].copy())
        return self.finished
    



    def episode_reset(self):
        """Reset controller state for a new episode."""
        self._plotted_once = False
        self._path_log = []
        self._tick = 0  # Important for timing and next episode

    def episode_callback(self, curr_time: float=None):
        """Callback at the end of each episode.
        
        Args:
            curr_time: Current simulation time.
        """
        pass

    def episode_callback(self, curr_time: float=None):
        """Callback at the end of each episode.
        
        Args:
            curr_time: Current simulation time.
        """
        pass




    def get_trajectory(self) -> NDArray[np.floating]:
        """Get the smoothed reference trajectory points.
        
        Returns:
            Array of shape (700, 3) containing interpolated trajectory points.
        """
        # Recreate spline interpolation of waypoints
        ts = np.linspace(0, 1, np.shape(self.waypoints)[0])
        cs_x = CubicSpline(ts, self.waypoints[:, 0])
        cs_y = CubicSpline(ts, self.waypoints[:, 1])
        cs_z = CubicSpline(ts, self.waypoints[:, 2])
        
        # Generate high-density points for visualization
        vis_s = np.linspace(0.0, 1.0, 700)
        traj_points = np.column_stack((cs_x(vis_s), cs_y(vis_s), cs_z(vis_s)))

        return traj_points
        
    def get_prediction_horizon(self) -> NDArray[np.floating]:
        """Get the predicted position trajectory for the planning horizon.
        
        Returns:
            Array of shape (N, 3) containing the predicted x,y,z positions
            for the next N timesteps in the planning horizon.
        """
        # Collect predicted states for all steps in the horizon
        horizon_positions = []
        for i in range(self.N):
            state = self.acados_ocp_solver.get(i, "x")
            # First three elements of the state are x,y,z positions
            pos = state[:3]
            horizon_positions.append(pos)
        
        return np.array(horizon_positions)
    


    def compute_min_distance_to_trajectory(self, drone_pos):
        """
        Berechnet den minimalen Abstand der Drohne zur Trajektorie,
        indem nur der Bereich hinter dem aktuellen Theta durchsucht wird.
        
        Args:
            drone_pos: Position der Drohne als np.array([x, y, z])
            
        Returns:
            tuple: (minimaler Abstand, Position des nächsten Punkts)
        """
        # Aktueller Referenzpunkt als obere Grenze der Suche
        current_theta = self.theta
        
        # Parameter für die Suche
        search_range = 0.1  # Wie weit zurück suchen (in theta-Einheiten)
        coarse_samples = 10  # Anzahl grober Samples für erste Suche
        fine_samples = 10     # Anzahl feiner Samples für die Verfeinerung
        
        # Definiere den Suchbereich (nur nach hinten)
        search_start = max(0.0, current_theta - search_range)
        search_end = current_theta  # Endet beim aktuellen Punkt
        
        # 1. Grobe Suche
        min_dist = float('inf')
        min_theta = current_theta
        
        # Überspringe die Suche, wenn wir am Anfang der Trajektorie sind
        if search_start >= search_end:
            # Berechne Abstand zum Anfangspunkt
            pos_x = self.cs_x(0)
            pos_y = self.cs_y(0)
            pos_z = self.cs_z(0)
            traj_pos = np.array([pos_x, pos_y, pos_z])
            return np.linalg.norm(drone_pos - traj_pos), traj_pos, 0
        
        # Grobe Abtastung des Suchbereichs
        theta_values = np.linspace(search_start, search_end, coarse_samples)
        for theta in theta_values:
            # Berechne Position auf der Trajektorie
            pos_x = self.cs_x(theta)
            pos_y = self.cs_y(theta)
            pos_z = self.cs_z(theta)
            traj_pos = np.array([pos_x, pos_y, pos_z])
            
            # Berechne Abstand
            dist = np.linalg.norm(drone_pos - traj_pos)
            
            # Aktualisiere Minimum
            if dist < min_dist:
                min_dist = dist
                min_theta = theta
                min_traj_pos = traj_pos
        
        # 2. Feine Suche
        fine_search_start = max(0.0, min_theta - search_range/coarse_samples)
        fine_search_end = min(current_theta, min_theta + search_range/coarse_samples)
        
        # Feine Abtastung des eingegrenzten Bereichs
        fine_theta_values = np.linspace(fine_search_start, fine_search_end, fine_samples)
        for theta in fine_theta_values:
            pos_x = self.cs_x(theta)
            pos_y = self.cs_y(theta)
            pos_z = self.cs_z(theta)
            traj_pos = np.array([pos_x, pos_y, pos_z])
            
            dist = np.linalg.norm(drone_pos - traj_pos)
            if dist < min_dist:
                min_dist = dist
                min_traj_pos = traj_pos
                min_theta = theta
        
        return min_dist, min_traj_pos, min_theta
    

    def get_weight(self, theta: float) -> float:
        """
        Gibt das Gewicht w(theta) zurück, das mindestens base_weight ist
        und an jedem Gate auf bis zu gate_peak_weights[i] ansteigt.
        """
        base_weight = 20
        sigma       = 0.05

        w = base_weight
        # Durchlaufe Gates und zugehörige Spitzengewichte
        for gθ, peak_w in zip(self.gate_thetas, self.gate_peak_weights):
            diff      = theta - gθ
            influence = (peak_w - base_weight) * np.exp(-0.5 * (diff/sigma)**2)
            w         = max(w, base_weight + influence)

        return w
    

    # =================== logger =========================
    def _log_xcurrent(self, t: float, xcurrent: np.ndarray) -> None:
        """Merke Zeit, xcurrent und Soll‑Position auf dem Pfad."""
        if self._tick % self._log_every == 0:
            ref_pos = np.array([self.cs_x(self.theta),
                                self.cs_y(self.theta),
                                self.cs_z(self.theta)], dtype=float)

            x_with_ref = np.hstack((xcurrent, ref_pos))
            self._xcurrent_log.append((t, x_with_ref))


    def get_xcurrent_log(self) -> list[tuple[float, np.ndarray]]:
        """Gibt das bisher gesammelte Log zurück (Liste von (t, xcurr))."""
        return self._xcurrent_log

    def reset_logs(self) -> None:
        """Leert alle internen Log‑Puffer – nach jedem Run aufrufen!"""
        self._xcurrent_log.clear()

    # =====================================================