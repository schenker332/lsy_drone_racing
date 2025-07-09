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
        self.N = 40                    # Number of discretization steps
        self.T_HORIZON = 1.5           # Time horizon in seconds
        self.dt = self.T_HORIZON / self.N  # Step size

        waypoints = np.array(
        [
            [1.0, 1.5, 0.2],
            [0.8, 1.0, 0.2],
            [0.7, 0.1, 0.4],
            [0.45, -0.5, 0.56],  # gate1
            [0.2, -0.7, 0.65],
            [0.1, -1.0, 0.75],
            [0.5, -1.5, 0.8],
            [1, -1.05, 1.1],    # gate2 1.1
            [1.15, -0.75, 1],
            [0.5, 0, 0.8],
            [0, 1, 0.56],        # gate3
            [-0.2, 1.4, 0.56],
            [-0.9, 1.3, 0.8], 
            [-0.5, 0, 1.11],     # gate4
	        [-0.4, -1, 1.11]
	        # old final point of max below, potential issue for shortcut
            #[-0.6, -2, 1.11] # point after gate for clean trajectory to finish 
            ### LEARNING: Last poun needs to be really far out for a smooth trajectory though the gate (at least with our current trajectory calculation)
        ])


        
        
        ts = np.linspace(0, 1, np.shape(waypoints)[0])
        self.cs_x = CubicSpline(ts, waypoints[:, 0])
        self.cs_y = CubicSpline(ts, waypoints[:, 1])
        self.cs_z = CubicSpline(ts, waypoints[:, 2])


        self.theta = 0
        self.v_theta = 1/ (7 * self.dt * self.freq) ## from niclas with 6 or 7
        gate_indices = [3, 7, 10, 13]
        self.gate_thetas = [ts[i] for i in gate_indices]
        ### LEARNING: Weights that are higher than 60 can cause huge issues, 
        # if the drone is not able to get the curve (i.e. reversing back, crashing into the gate, 
        # or deliberately missting the gate to avoid a high off-center penalty)
        self.gate_peak_weights = [100, 80, 60, 140] # [40, 80, 60, 140]. //// [140, 140, 140, 140] 
	


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

        self._last_gates_visited = None  # Initialize gate tracking
        self._last_gates_pos = None      # Store previous gate positions for delta calculation
        
        self.gate_to_waypoint_mapping = {
            0: 3,   # Gate 0 -> Waypoint 3
            1: 7,   # Gate 1 -> Waypoint 7  
            2: 10,   # Gate 2 -> Waypoint 10
            3: 13,   # Gate 3 -> Waypoint 13

        }

        # Mapping Gate -> Liste von (waypoint_idx, factor) Tupeln,
        # wobei factor bestimmt, wie stark die Δ-Verschiebung angewendet wird
        # factor = 1.0 bedeutet volle Verschiebung, 0.5 bedeutet halbe Verschiebung, etc.
        self.relative_waypoint_mapping = {
            # 3: [(14, 1.0)],  # Gate 3 verschiebt Waypoint 14 mit voller Stärke
            # 0: [(7, 2)],  # Gate 1 verschiebt WP 6 mit 50% und WP 8 mit 30% der Delta-Verschiebung
            # Beispiele für weitere Optionen:
            # 1: [(6, 0.5), (8, 0.3)],  # Gate 1 verschiebt WP 6 mit 50% und WP 8 mit 30% der Delta-Verschiebung
            # hier beliebig weitere Einträge möglich
        }

        # Gate-Offsets für feinere Kontrolle der Flugbahn
        # Format: [dx, dy, dz] in Meter
        self.gate_offsets = {
            0: np.array([0.0, 0.0, 0.0]),    # Kein Offset für Gate 0
            1: np.array([0.0, 0.0, 0.0]),    # Kein Offset für Gate 1
            2: np.array([0.0, 0.0, 0.0]),    # Offset für Gate 2: 20cm rechts, 10cm höher
            3: np.array([0.0, 0.0, 0.0])   # Offset für Gate 3: 10cm links, 15cm höher
        }

        # for visualization 
        vis_s = np.linspace(0.0, 1.0, 700)
        self._trajectory_history = []  # Store all trajectories for visualization
        temp = np.column_stack((self.cs_x(vis_s), self.cs_y(vis_s), self.cs_z(vis_s)))
        self._trajectory_history.append(temp.copy())  





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



        ### ======================================= ###
        ### ============ Set parameters =========== ###
        ### ======================================= ###

        _,_,min_theta = self.compute_min_distance_to_trajectory(obs["pos"])

        # update trajectory
        self._handle_gate_update(obs)
        obs_array = obs["obstacles_pos"]          # shape=(4,3)
        obs_flat = obs_array.reshape(-1)




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
            p_ref = np.concatenate([p_ref, obs_flat])
            self.acados_ocp_solver.set(j, "p", p_ref)


        ### ======================================= ###
        ### ============ Solve the OCP ============ ###
        ### ======================================= ###
        self.acados_ocp_solver.solve()
        x1 = self.acados_ocp_solver.get(1, "x")


        ### ======================================= ###
        ### ============ Debugging ================ ###
        ### ======================================= ###

        # Bestrafungsparameter
        weight = 40.0   # Gewicht
        k      = 5.0   # Steilheit

        # Beispiel: obs_flat aus deinem Controller, shape (12,)
        # Und obs["pos"] aus deinem Sensor-Input, shape (3,)
        drone_xy = np.array(obs["pos"][:2])        # [px, py]
        obs_flat  = obs["obstacles_pos"].reshape(-1)  # (12,)

        # Umformen zu (4,3) und nur x,y behalten → (4,2)
        obs_xy = obs_flat.reshape(-1, 3)[:, :2]     

        # Loop über alle Hindernisse
        for i, (ox, oy) in enumerate(obs_xy):
            # Abstand in der xy-Ebene
            dx = drone_xy[0] - ox
            dy = drone_xy[1] - oy
            d  = np.hypot(dx, dy)                  # = sqrt(dx**2 + dy**2)

            # Exponentielle Kosten
            cost_i = weight * np.exp(-k * d)

            # print(f"Obs {i}: Position=({ox:6.3f}, {oy:6.3f})  Abstand={d:5.3f} m  Kosten={cost_i:6.3f}")
        # print_output(tick=self._tick, obs=obs, freq=self.config.env.freq)

        #u1 = self.acados_ocp_solver.get(1, "u")
        # Debugging of  feedback law i.e print u1
        # # print u1
        # input_names = ["df_cmd", "dr_cmd", "dp_cmd", "dy_cmd", "dv_theta_cmd"]
        # for name, value in zip(input_names, u1):
        #     print(f"{name}: {value:.19f}")

	    ## Debugging prints for state variables
        # # print_output(tick=self._tick, obs=obs, freq=self.config.env.freq)
        # state_names = ["px", "py", "pz", "vx", "vy", "vz", "roll", "pitch", "yaw",
        #                "f_collective", "f_collective_cmd", "r_cmd", "p_cmd", "y_cmd",
        #                "theta", "v_theta"]

        # for name, value in zip(state_names, x1):
        #     print(f"{name}: {value}")
        
        # print("=" * 20)
        
        # # print(f"weight: {self.get_weight(min_theta):.2f}")
        # print(f"min_dist: {min_dist:.2f} at theta: {min_theta:.2f}")
        # print(f"Time: {self._tick/self.freq:.2f}s")
        # print("=" * 20)


        ### ======================================= ###
        ### ============ Update state ============= ###
        ### ======================================= ###

        w = 1 / self.config.env.freq / self.dt
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]  

        self.v_theta = x1[15] 
        self.theta = min(1.0, self.theta + self.v_theta * self.dt)
        #self.theta = x1[14]  # Experimental Update theta directly from the MPC solution

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




    def get_trajectory(self) -> NDArray[np.floating]:
        return self._trajectory_history
        
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
        sigma       = 0.06

        w = base_weight
        # Durchlaufe Gates und zugehörige Spitzengewichte
        for gθ, peak_w in zip(self.gate_thetas, self.gate_peak_weights):
            diff      = theta - gθ
            influence = (peak_w - base_weight) * np.exp(-0.5 * (diff/sigma)**2)
            w         = max(w, base_weight + influence)

        return w
    
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


    def _handle_gate_update(self, obs: dict[str, NDArray[np.floating]]):
            """Checks for gate changes and updates the trajectory spline if a new gate is passed."""
            gates_pos = obs.get("gates_pos", None)
            gates_visited = obs.get("gates_visited", None)

            # Gate change detection
            if self._last_gates_visited is not None and gates_visited is not None:
                if not np.array_equal(gates_visited, self._last_gates_visited):
                    # Find which specific gate changed
                    for i, (old, new) in enumerate(zip(self._last_gates_visited, gates_visited)):
                        if old != new and gates_pos is not None and i < len(gates_pos):
                            # --- 1) Berechne neue und alte finale Gate-Position inkl. Offset ---
                            new_gate_pos = gates_pos[i].copy()
                            old_gate_pos = (self._last_gates_pos[i]
                                            if self._last_gates_pos is not None
                                            else new_gate_pos)
                            
                            # Offset-Logik für neue Position
                            if i in self.gate_offsets:
                                gates_orn = obs.get("gates_orn", None)
                                if gates_orn is not None and i < len(gates_orn):
                                    # Anwenden der Rotation auf den Offset
                                    offset_local = self.gate_offsets[i]
                                    rot = R.from_quat(gates_orn[i])
                                    offset_global = rot.apply(offset_local)
                                    new_gate_pos += offset_global
                                    # Auch alte Position mit gleichem Offset für korrekte Delta-Berechnung
                                    old_gate_pos += offset_global
                                else:
                                    # Wenn keine Orientierung verfügbar, einfachen Offset anwenden
                                    new_gate_pos += self.gate_offsets[i]
                                    old_gate_pos += self.gate_offsets[i]
                            
                            # Δ-Verschiebung berechnen
                            delta = new_gate_pos - old_gate_pos
                            
                            # --- 2) Haupt-Waypoint für dieses Gate updaten ---
                            if i in self.gate_to_waypoint_mapping:
                                waypoint_idx = self.gate_to_waypoint_mapping[i]
                                self.waypoints[waypoint_idx] = new_gate_pos
                            
                            # --- 3) Alle in relative_waypoint_mapping definierten Waypoints updaten ---
                            if i in self.relative_waypoint_mapping:
                                for wp_rel, factor in self.relative_waypoint_mapping[i]:
                                    # Prüfe Index-Gültigkeit
                                    if 0 <= wp_rel < len(self.waypoints):
                                        # Wende die Verschiebung mit dem angegebenen Faktor an
                                        self.waypoints[wp_rel] += delta * factor
                            
                            # --- 4) Splines neu berechnen ---
                            ts = np.linspace(0, 1, np.shape(self.waypoints)[0])
                            self.cs_x = CubicSpline(ts, self.waypoints[:, 0])
                            self.cs_y = CubicSpline(ts, self.waypoints[:, 1])
                            self.cs_z = CubicSpline(ts, self.waypoints[:, 2])
                            
                            # for visualization
                            vis_s = np.linspace(0.0, 1.0, 700)
                            new_traj = np.column_stack((self.cs_x(vis_s), self.cs_y(vis_s), self.cs_z(vis_s)))
                            self._trajectory_history.append(new_traj.copy())

            # Store for next comparison
            self._last_gates_visited = gates_visited.copy() if gates_visited is not None else None
            self._last_gates_pos = gates_pos.copy() if gates_pos is not None else None

