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


# Sigma defines how much we stretch the Peak weights at each gate
GATE_WEIGHT_CONFIG = {
    0: {"peak_weight": 35, "sigma": 0.04},  # Gate 0: narrow peak
    1: {"peak_weight": 50, "sigma": 0.02},  # Gate 1: wider peak
    2: {"peak_weight": 60, "sigma": 0.08},  # Gate 2: narrow peak
    3: {"peak_weight": 120, "sigma": 0.04}  # Gate 3: very narrow, high peak
}

# self.gate_peak_weights = [35, 50, 60, 120] 

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
        ### ====================================================================== ###
        ### =========================== Waypoints ================================ ###
        ### ====================================================================== ###

        # Gate-Posittions better for simulation
        gates = np.array([
            [0.45, -0.50, 0.56],  # Gate 0      [0.45, -0.5, 0.56]
            [1.0, -1.05, 1.2],    # Gate 1      [1.0, -1.05, 1.11]
            [-0.04, 1.0, 0.59],   # Gate 2      [0.0, 1.0, 0.56]       #     [0.04, 1.0, 0.66],  (real deployment) 
            [-0.5, 0.0, 1.11]     # Gate 3      [-0.5, 0.0, 1.11]
        ])
        
        
        # Start to Gate 0
        b0 = np.array([
            [1.0, 1.5, 0.2], #[1.0, 1.5, 0.15]
            [0.8, 1.0, 0.2], #[0.85, 1.0, 0.2]
            [0.7, 0.1, 0.4]
        ])

        
        # Gate 0 -- Gate 1
        b1 = np.array([
            [0.25, -0.7, 0.65],  # [0.2, -0.7, 0.65],
            # [0.1, -1.0, 0.75],
            [0.5, -1.5, 0.8]
        ])
        
        # Gate 1 -- Gate 2
        b2 = np.array([
            [1.15, -0.75, 1],
            [0.5, 0, 0.8]
        ])
        
        # Gate 2 -- Gate 3
        b3 = np.array([
            [-0.2, 1.4, 0.66],
            [-0.97, 1.3, 0.8],      #[-0.9, 1.3, 0.8],
            #[-0.75, 0.5, 1.11]   
            # -                     #[-0.85, 0.5, 0.9]  
        ])


        
        # after Gate 3 
        b4 = np.array([
            [-0.1, -1, 1.2]
        ])
        
        # Kombiniere alle Waypoints: b0 + gate0 + b1 + gate1 + b2 + gate2 + b3 + gate3 + b4
        waypoint_sections = [b0, gates[0:1], b1, gates[1:2], b2, gates[2:3], b3, gates[3:4], b4]
        waypoints = np.vstack(waypoint_sections)
        
        # Automatische Indizes berechnen
        current_index = 0
        gate_indices = []
        waypoint_blocks = {}
        
        for i, section in enumerate(waypoint_sections):
            section_length = len(section)
            
            if i % 2 == 1:  # Ungerade Indizes sind Gates (1, 3, 5, 7)
                gate_idx = i // 2  # 0, 1, 2, 3
                gate_indices.append(current_index)
            else:  # Gerade Indizes sind Waypoint-Blöcke (0, 2, 4, 6, 8)
                block_idx = i // 2  # 0, 1, 2, 3, 4
                waypoint_blocks[block_idx] = list(range(current_index, current_index + section_length))
            
            current_index += section_length
        
        # Store strukturierte Daten
        self.waypoint_blocks = waypoint_blocks
        self.gate_indices = gate_indices
        self.waypoints = waypoints

        
        ts = np.linspace(0, 1, np.shape(waypoints)[0])
        self.cs_x = CubicSpline(ts, waypoints[:, 0])
        self.cs_y = CubicSpline(ts, waypoints[:, 1])
        self.cs_z = CubicSpline(ts, waypoints[:, 2])


        ### ====================================================================== ###
        ### =========================== Mapping ================================== ###
        ### ====================================================================== ###

        
        # Automatisches Gate-zu-Waypoint-Mapping
        self.gate_to_waypoint_mapping = {i: gate_indices[i] for i in range(len(gate_indices))}

        
        self.response_mapping = {
            "g0": [  # Wenn sich Gate 0 ändert
            ("g0", 1.3, 1.3, 1.3, [0.075, 0.0, 0.0]),  # Gate 0 selbst mit Offset
            ],

            "g1": [  # Wenn sich Gate 1 ändert  
            ("g1", 1.0, 1.0, 1.0, [0.0, 0.0, 0.0]),    # Gate 1 selbst mit Offset
            ],

            "g2": [  # Wenn sich Gate 2 ändert
            ("g2", 1.0, 1.0, 1.0, [0.0, 0.0, 0.0]),    # Gate 2 selbst (kein Offset)
            ("b2.0", 0.3, 0.5, 0.0, [0.0, 0.0, 0.0]),  # Block 2, Waypoint 0
            ],

            "g3": [  # Wenn sich Gate 3 ändert
            ("g3", 1.0, 1.0, 1.0, [0.0, 0.0, 0.0]),    # Gate 3 selbst mit Offset
            ("b4.0", 1.0, 1.0, 1.0, [0.0, 0.0, 0.0]),  # Block 3, Waypoint 0 (at obstacle 1)
            ],

            "o1": [  # Wenn sich Obstacle 1 ändert  
            ("b1.0", 1.0, 1.0, 0.0, [0.0, 0.0, 0.0]),  # Block 1, Waypoint 0 (at obstacle 2)
            ("b1.1", 1.0, 1.0, 0.0, [0.0, 0.0, 0.0]),  # Block 1, Waypoint 1
            ("b1.2", 1.0, 1.0, 0.0, [0.0, 0.0, 0.0]),  # Block 1, Waypoint 2
            ],

            #add reaction to obstacle 2 from top of tree ?

            "o3": [  # Wenn sich Obstacle 3 ändert
            ("b3.2", 2.0, 2.0, 0.0, [0.0, 0.0, 0.0]),  # Block 3, Waypoint 2 (at obstacle 4)
            ],
        }
        
        self._last_obstacles_pos = None  # Store previous obstacle positions

        # for visualization 
        vis_s = np.linspace(0.0, 1.0, 700)
        self._trajectory_history = []  # Store all trajectories for visualization
        temp = np.column_stack((self.cs_x(vis_s), self.cs_y(vis_s), self.cs_z(vis_s)))
        self._trajectory_history.append(temp.copy())  


        ### ====================================================================== ###
        ### ============================ Settings ================================ ###
        ### ====================================================================== ###

        self.freq = config.env.freq
        self._tick = 0

        # MPC parameters
        self.N = 50                   #50
        self.T_HORIZON = 1.5           #1.5
        self.dt = self.T_HORIZON / self.N  # Step size

        # Initialize state variables    
        self.last_f_collective = 0.3  # Current thrust value
        self.last_f_cmd = 0.3     # Commanded thrust value
        self.last_rpy_cmd = np.zeros(3)  # Commanded roll, pitch, yaw

        # Store configuration and tracking variables
        self.config = config
        self.finished = False
        self._info = info
        self._path_log = []

        self._last_gates_visited = None  # Initialize gate tracking
        self._last_gates_pos = None      # Store previous gate positions for delta calculation


        self.theta = 0.0  # Current theta value (progress along trajectory)
        t= 3.6
        self.v_theta = 1/ (t * self.dt * self.freq) ## from niclas with 6 or 7



        self.gate_thetas = [ts[i] for i in gate_indices]


        self.gate_peak_weights = [GATE_WEIGHT_CONFIG[i]["peak_weight"] for i in range(4)]
        self.gate_sigmas = [GATE_WEIGHT_CONFIG[i]["sigma"] for i in range(4)]



        # Create the optimal control problem solver
        self.acados_ocp_solver, self.ocp = create_ocp_solver(self.T_HORIZON, self.N)

        self.base_weight = 70   
        
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

        self.pos = obs["pos"]
        # update trajectory
        self._handle_unified_update(obs)



        # Prepare reference trajectory and weights for all steps in the horizon
        for j in range(self.N + 1):
            theta_j = min(self.theta + j * self.v_theta * self.dt, 1.0)
            theta_j_next = min(theta_j + 0.0001, 1.0)
            theta_min_j = min(min_theta + j * self.v_theta * self.dt, 1.0)


            p_ref = np.array([
            self.cs_x(theta_j), self.cs_y(theta_j), self.cs_z(theta_j), # for contouring
            self.cs_x(theta_j_next), self.cs_y(theta_j_next), self.cs_z(theta_j_next), #for contouring
            self.weight_from_theta(theta_min_j),   # for gauss weighting
            self.cs_x(theta_min_j), self.cs_y(theta_min_j), self.cs_z(theta_min_j) # for real minimun distance
            ])
            self.acados_ocp_solver.set(j, "p", p_ref)


        ### ========== logger ==========================
        curr_sim_time = self._tick / self.freq   # reale Simulationszeit (s)
        self._log_xcurrent(curr_sim_time, xcurrent)
        ### ============================================


        ### ======================================= ###
        ### ============ Solve the OCP ============ ###
        ### ======================================= ###
        self.acados_ocp_solver.solve()
        x1 = self.acados_ocp_solver.get(1, "x")
        u1 = self.acados_ocp_solver.get(1, "u")




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

 



    # def get_trajectory(self) -> NDArray[np.floating]:
    #     """Get the smoothed reference trajectory points.
        
    #     Returns:
    #         Array of shape (700, 3) containing interpolated trajectory points.
    #     """
    #     # Recreate spline interpolation of waypoints
    #     ts = np.linspace(0, 1, np.shape(self.waypoints)[0])
    #     cs_x = CubicSpline(ts, self.waypoints[:, 0])
    #     cs_y = CubicSpline(ts, self.waypoints[:, 1])
    #     cs_z = CubicSpline(ts, self.waypoints[:, 2])
        
    #     # Generate high-density points for visualization
    #     vis_s = np.linspace(0.0, 1.0, 700)
    #     traj_points = np.column_stack((cs_x(vis_s), cs_y(vis_s), cs_z(vis_s)))

    #     return traj_points
        
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
    





    def get_trajectory(self) -> NDArray[np.floating]:
        return self._trajectory_history
    
    def get_waypoints(self) -> NDArray[np.floating]:
        return {
            'waypoints': self.waypoints,
            'gate_indices': self.gate_indices,
            'waypoint_blocks': self.waypoint_blocks,
            'gate_to_waypoint_mapping': self.gate_to_waypoint_mapping
        }

    # def get_prediction_horizon(self) -> NDArray[np.floating]:
    #     """Get the predicted position trajectory for the planning horizon.
        
    #     Returns:
    #         Array of shape (N, 3) containing the predicted x,y,z positions
    #         for the next N timesteps in the planning horizon.
    #     """
    #     # Collect predicted states for all steps in the horizon
    #     horizon_positions = []
    #     for i in range(self.N):
    #         state = self.acados_ocp_solver.get(i, "x")
    #         # First three elements of the state are x,y,z positions
    #         pos = state[:3]
    #         horizon_positions.append(pos)
        
    #     return np.array(horizon_positions)


    
    def get_visualization_data(self, drone_pos):
        """Calculate visualization data for plotting error vectors and reference points.
        
        Args:
            drone_pos: Current drone position as np.array([x, y, z])
            
        Returns:
            dict: Dictionary containing all visualization data with keys:
                - ref_point: Reference point on trajectory
                - next_point: Next point for tangent calculation  
                - t_hat: Unit tangent vector
                - e_vec: Error vector (drone_pos - ref_point)
                - e_l_scalar: Lag error scalar
                - e_l_vec: Lag error vector
                - e_c_vec: Contour error vector
                - t_hat_scaled: Scaled tangent vector for visualization
                - e_l_vis: Lag error visualization point
                - e_c_vis: Contour error visualization point
        """
        # Get current theta reference point on trajectory
        theta = self.theta
        ref_x = self.cs_x(theta)
        ref_y = self.cs_y(theta)
        ref_z = self.cs_z(theta)
        ref_point = np.array([ref_x, ref_y, ref_z])

        # Calculate next point for tangent vector
        theta_next = theta + 0.0001
        next_x = self.cs_x(theta_next)
        next_y = self.cs_y(theta_next)
        next_z = self.cs_z(theta_next)
        next_point = np.array([next_x, next_y, next_z])

        # Calculate unit tangent vector t_hat
        tangent = next_point - ref_point
        t_hat = tangent / (np.linalg.norm(tangent) + 1e-10)
        
        # Error vector e = pos - ref
        e_vec = drone_pos - ref_point
        
        # Calculate Contour Error (e_c) and Lag Error (e_l)
        e_l_scalar = np.dot(t_hat, e_vec)  # Projection onto tangent
        e_l_vec = e_l_scalar * t_hat        # Vector in tangent direction
        e_c_vec = e_vec - e_l_vec           # Orthogonal vector
        
        # Scale vectors for better visualization
        scale_factor = 1
        t_hat_scaled = ref_point + t_hat * scale_factor
        e_l_vis = ref_point + e_l_vec
        e_c_vis = ref_point + e_c_vec
        
        return {
            'ref_point': ref_point,
            'next_point': next_point,
            't_hat': t_hat,
            'e_vec': e_vec,
            'e_l_scalar': e_l_scalar,
            'e_l_vec': e_l_vec,
            'e_c_vec': e_c_vec,
            't_hat_scaled': t_hat_scaled,
            'e_l_vis': e_l_vis,
            'e_c_vis': e_c_vis
        }







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
    

    # def get_weight(self, theta: float) -> float:
    #     """
    #     Gibt das Gewicht w(theta) zurück, das mindestens base_weight ist
    #     und an jedem Gate auf bis zu gate_peak_weights[i] ansteigt.
    #     """
    #     base_weight = 20
    #     sigma       = 0.05

    #     w = base_weight
    #     # Durchlaufe Gates und zugehörige Spitzengewichte
    #     for gθ, peak_w in zip(self.gate_thetas, self.gate_peak_weights):
    #         diff      = theta - gθ
    #         influence = (peak_w - base_weight) * np.exp(-0.5 * (diff/sigma)**2)
    #         w         = max(w, base_weight + influence)

    #     return w
    

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

    def weight_from_theta(self, theta: float) -> float:
        """
        Calculates the contouring cost weight based on the progress `theta`.

        The weight is increased when the drone is near a gate, creating a
        Gaussian-like peak in the cost function. This encourages tighter
        tracking in critical regions of the trajectory.

        Args:
            theta: The current progress along the trajectory (0 to 1).

        Returns:
            The calculated weight for the contouring cost.
        """
        # Start with the base weight (default cost away from gates)
        w = self.base_weight
        # Loop over all gates with their individual sigmas and peak weights
        for gate_theta, peak_weight, sigma in zip(self.gate_thetas, self.gate_peak_weights, self.gate_sigmas):
            # Compute the normalized distance from the current theta to the gate's theta
            diff = (theta - gate_theta) / sigma  # self.sigma controls the width of the peak
            # Compute the Gaussian-shaped influence of this gate
            gate_influence = (peak_weight - self.base_weight) * np.exp(-0.5 * diff * diff)
            # The weight at this theta is the maximum of the base weight and all gate influences
            w = max(w, self.base_weight + gate_influence)
        # Return the final weight for this theta
        return w
    
    def _handle_unified_update(self, obs: dict[str, NDArray[np.floating]]):
        """Handles both gate and obstacle changes with unified response mapping."""
        
        # Handle Gate Changes
        gates_pos = obs.get("gates_pos", None)
        gates_visited = obs.get("gates_visited", None)

        
        if self._last_gates_visited is not None and gates_visited is not None and gates_pos is not None:
            if not np.array_equal(gates_visited, self._last_gates_visited):
                for i, (old, new) in enumerate(zip(self._last_gates_visited, gates_visited)):
                    if old != new and i < len(gates_pos):
                        old_pos = self._last_gates_pos[i] if self._last_gates_pos is not None else gates_pos[i]
                        # Nur response_mapping anwenden, keine automatische Gate-Positionierung
                        self._apply_unified_response(f"g{i}", gates_pos[i], old_pos)


        
        # Handle Obstacle Changes  
        obstacles_pos = obs.get("obstacles_pos", None)
        
        if obstacles_pos is not None and self._last_obstacles_pos is not None:
            for i, (old_pos, new_pos) in enumerate(zip(self._last_obstacles_pos, obstacles_pos)):
                if np.linalg.norm(new_pos - old_pos) > 0.001:
                    self._apply_unified_response(f"o{i}", new_pos, old_pos)


        
        # Store for next comparison
        self._last_gates_visited = gates_visited.copy() if gates_visited is not None else None
        self._last_gates_pos = gates_pos.copy() if gates_pos is not None else None
        self._last_obstacles_pos = obstacles_pos.copy() if obstacles_pos is not None else None

    def _apply_unified_response(self, trigger_id: str, new_pos: np.ndarray, old_pos: np.ndarray):
        """Apply unified response mapping for any trigger (gate or obstacle change)."""
        
        if trigger_id not in self.response_mapping:
            return
        
        delta = new_pos - old_pos
    
        trajectory_updated = False 
        for target, x_factor, y_factor, z_factor, offset in self.response_mapping[trigger_id]:
            
            # Calculate response with factors and offset
            response_delta = np.array([
                delta[0] * x_factor + offset[0],
                delta[1] * y_factor + offset[1], 
                delta[2] * z_factor + offset[2]
            ])
            
            # Apply to target
            if target.startswith("g"):  # Gate target
                gate_idx = int(target[1:])
                if gate_idx in self.gate_to_waypoint_mapping:
                    wp_idx = self.gate_to_waypoint_mapping[gate_idx]
                    if 0 <= wp_idx < len(self.waypoints):
                        self.waypoints[wp_idx] += response_delta
                        trajectory_updated = True
                        
            elif target.startswith("b"):  # Block waypoint target
                # Parse "b2.1" -> block=2, waypoint_in_block=1
                parts = target[1:].split(".")
                block_idx = int(parts[0])
                wp_in_block = int(parts[1])
                
                if block_idx in self.waypoint_blocks:
                    if wp_in_block < len(self.waypoint_blocks[block_idx]):
                        wp_idx = self.waypoint_blocks[block_idx][wp_in_block]
                        if 0 <= wp_idx < len(self.waypoints):
                            self.waypoints[wp_idx] += response_delta
                            trajectory_updated = True
        
        # Rebuild splines if any waypoint was updated
        if trajectory_updated:
            ts = np.linspace(0, 1, len(self.waypoints))
            self.cs_x = CubicSpline(ts, self.waypoints[:, 0])
            self.cs_y = CubicSpline(ts, self.waypoints[:, 1])
            self.cs_z = CubicSpline(ts, self.waypoints[:, 2])
            
            # Visualization update
            vis_s = np.linspace(0.0, 1.0, 700)
            new_traj = np.column_stack((self.cs_x(vis_s), self.cs_y(vis_s), self.cs_z(vis_s)))
            self._trajectory_history.append(new_traj.copy())


