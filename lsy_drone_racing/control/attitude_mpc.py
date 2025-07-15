from __future__ import annotations 
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

from lsy_drone_racing.control.create_ocp_solver import create_ocp_solver
from lsy_drone_racing.control.helper.datalogger import DataLogger
import pathlib
import subprocess
import sys



if TYPE_CHECKING:
    from numpy.typing import NDArray


class MPController(Controller):
    """Model Predictive Controller using collective thrust and attitude interface.

    This controller implements a Model Predictive Control (MPC) strategy to pilot a
    quadrotor drone through a series of gates. It uses a collective thrust and
    attitude command interface, where the controller computes the desired total
    thrust and orientation (roll, pitch, yaw) for the drone.

    The controller follows a pre-defined trajectory, represented by a cubic spline,
    and uses an MPC solver to optimize the drone's path over a finite time horizon.
    The cost function is designed to minimize contouring and lag errors with respect
    to the reference trajectory, while also penalizing excessive control inputs.

    Key features include:
    - Waypoint-based trajectory generation.
    - Dynamic adjustment of trajectory tracking aggressiveness based on proximity to gates.
    - Logging of state, control, and reference data for analysis.
    """

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

        # # Adapted to niclas trajectory Gate-Positionen
        # gates = np.array([
        #     [0.45, -0.50, 0.56],  # Gate 0      [0.45, -0.5, 0.56]
        #     [1.0, -1.05, 1.25],   # Gate 1      [1.0, -1.05, 1.11]
        #     [0.0, 1.0, 0.56],     # Gate 2      [0.0, 1.0, 0.56]
        #     [-0.5, 0.0, 1.11]     # Gate 3      [-0.5, 0.0, 1.11]
        # ])

        ### Original Gate-Positions
        gates = np.array([
            [0.45, -0.5, 0.56],     # Gate 0      
            [1.0, -1.05, 1.11],     # Gate 1     
            [0.0, 1.0, 0.56],       # Gate 2      
            [-0.5, 0.0, 1.11]       # Gate 3     
        ]) 
        
        # Start bis Gate 0
        b0 = np.array([
            [1.0, 1.5, 0.2],
            [0.95, 1.0, 0.3],
            [0.7, 0.1, 0.4],
        ])
        
        # Gate 0 -- Gate 1
        b1 = np.array([
            [0.2, -1.4, 0.85],
        ])
        
        # Gate 1 -- Gate 2
        b2 = np.array([
            [1.15, -0.75, 1],
            [0.65, -0.25, 0.85],
            # [0.4, 0, 0.8]
        ])
        
        # Gate 2 -- Gate 3
        b3 = np.array([
            [-0.2, 1.5, 0.56],
            [-0.9, 1.3, 0.8],
            [-0.75, 0.5, 1.11],     
        ])
        
        # nach Gate 3 
        b4 = np.array([
            [-0.5, -2, 1.11],
            [-0.5, -6, 1.11],
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

        # ===== EINHEITLICHES RESPONSE-MAPPING =====
        # Format: "trigger": [("target", x_factor, y_factor, z_factor, [offset_x, offset_y, offset_z])]
        # Trigger: "g0", "g1", "g2", "g3" für Gates oder "o0", "o1", "o2", "o3" für Obstacles
        # Target: "g0", "g1", etc. für Gates oder "b0.1", "b1.0", etc. für Block-Waypoints
        
        self.response_mapping = {
            "g0": [  # Wenn sich Gate 0 ändert
            ("g0", 1.3, 1.3, 1.3, [0.0, 0.0, 0.0]),  # Gate 0 selbst mit Offset
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
            ],

            "o1": [  # Wenn sich Obstacle 1 ändert  
            ("b1.0", 1.0, 1.0, 0.0, [0.0, 0.0, 0.0]),  # Block 1, Waypoint 0 (at obstacle 2)
            ("b1.1", 1.0, 1.0, 0.0, [0.0, 0.0, 0.0]),  # Block 1, Waypoint 1
            ("b1.2", 1.0, 1.0, 0.0, [0.0, 0.0, 0.0]),  # Block 1, Waypoint 2
            ],

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
        # --- MPC and Simulation Settings ---
        self.freq = config.env.freq  # Control frequency
        self._tick = 0  # Simulation tick counter

        # Toggle logging by setting this flag to True or False
        self.logging_enabled = True
        if  self.logging_enabled:
            # Initialize logger
            self.logger = DataLogger(log_dir="logs")
            self._last_log_time = -1
        else:
            self.logger = None

        # --- MPC Parameters ---
        self.N = 20  # Number of discretization steps in the horizon
        self.T_HORIZON = 0.6  # Time horizon in seconds
        self.dt = self.T_HORIZON / self.N  # Step size

        # --- Initial State Variables ---
        self.last_f_collective = 0.3  # Initial collective thrust
        self.last_f_cmd = 0.3  # Initial commanded thrust
        self.last_rpy_cmd = np.zeros(3)  # Initial commanded roll, pitch, yaw rates

        # --- Tracking and Path Variables ---
        self.config = config #set the configuration
        self.finished = False  # Flag to indicate if the trajectory is completed
        self._info = info
        self._path_log = []
        self._last_gates_visited = None
        self._last_gates_pos = None

        # --- Trajectory Progress ---
        self.theta = 0.0  # Progress along the trajectory (0 to 1)
        # Time to complete the trajectory, affects the base progress speed
        t = 4.7
        self.v_theta = 1 / (t * self.dt * self.freq)  # Base rate of progress
        self.base_v_theta = self.v_theta

        # --- Curvature Calculation ---
        # Derivatives of the spline for curvature calculation
        dx = self.cs_x.derivative(1)
        ddx = self.cs_x.derivative(2)
        dy = self.cs_y.derivative(1)
        ddy = self.cs_y.derivative(2)
        dz = self.cs_z.derivative(1)
        ddz = self.cs_z.derivative(2)

        def curvature(theta):
            """
            Calculate the curvature of the trajectory at a given progress `theta`.

            The curvature is a measure of how sharply the trajectory bends at a specific point.
            It is computed using the first and second derivatives of the trajectory with respect to theta.

            Args:
                theta: The progress parameter along the trajectory (0 to 1).

            Returns:
                The scalar curvature value at the given theta.
            """
            # v: First derivative (velocity vector) of the trajectory at theta
            v = np.array([dx(theta), dy(theta), dz(theta)])
            # a: Second derivative (acceleration vector) of the trajectory at theta
            a = np.array([ddx(theta), ddy(theta), ddz(theta)])
            # Numerator: Magnitude of the cross product of v and a
            num = np.linalg.norm(np.cross(v, a))
            # Denominator: Cube of the norm of the velocity vector

            den = np.linalg.norm(v)**3 + 1e-8
            # Curvature formula: |v x a| / |v|^3
            return num / den
        
        self.curvature = curvature    
        self.base_v_theta = self.v_theta

        # --- Gate-Specific Weighting ---
        # Find the `theta` values corresponding to each gate
        self.gate_thetas = [ts[i] for i in gate_indices]
        # Peak weights for the Gaussian-like cost function around each gate
        self.gate_peak_weights = [200, 200, 200, 300] # [40, 80, 60, 140]. //// [140, 140, 140, 140] 

        # --- OCP Solver Setup ---
        self.acados_ocp_solver, self.ocp = create_ocp_solver(self.T_HORIZON, self.N)

        # --- Contouring Cost Weighting ---
        # Base weight for the contouring error cost
        self.base_weight = 130.0
        # Sigma for the Gaussian-like weight distribution around gates
        self.sigma = 0.01 # 5% of the norm scale

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        This method is the main control loop. It constructs the current state,
        sets the MPC parameters, solves the OCP, and returns the computed control command.

        Args:
            obs: Current observation of the environment's state.
            info: Optional additional information.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        if self.theta >= 1.2:
            self.finished = True

        # Construct the current state vector for the MPC solver
        xcurrent = np.concatenate(
            (
                obs["pos"],
                obs["vel"],
                R.from_quat(obs["quat"]).as_euler("xyz", degrees=False),
                [self.last_f_collective, self.last_f_cmd],
                self.last_rpy_cmd,
                [self.theta, self.v_theta],
            )
        )

        # Log state vector periodically
        if self.logger:
            current_time = self._tick / self.freq
            if current_time - self._last_log_time >= 0.01:
                # Calculate current curvature and distance
                kappa = self.curvature(self.theta)
                min_dist, _, _ = self.compute_min_distance_to_trajectory(obs["pos"], self.theta)
                
                # Referenzpunkt auf der Trajektorie
                ref_pt = np.array([
                    self.cs_x(self.theta),
                    self.cs_y(self.theta),
                    self.cs_z(self.theta)
                ])
                
                # Get current weight (this is what gets stored in p[6])
                current_weight = self.weight_for_theta(self.theta)
                
                # Use existing get_visualization_data function for error calculation
                vis_data = self.get_visualization_data(obs["pos"])
                e_contour_magnitude = np.linalg.norm(vis_data['e_c_vec'])
                e_lag_magnitude = abs(vis_data['e_l_scalar'])
                
                try:
                    u1 = self.acados_ocp_solver.get(1, "u")
                    self.logger.log_state(
                        current_time, xcurrent, u1, 
                        ref_point=ref_pt, 
                        curvature=kappa, 
                        min_dist=min_dist
                    )
                except:
                    self.logger.log_state(
                        current_time, xcurrent, 
                        ref_point=ref_pt, 
                        curvature=kappa, 
                        min_dist=min_dist
                    )
                
                # Log weight data with actual error values
                self.logger.log_weight_data(
                    current_time, 
                    current_weight,
                    e_contour=e_contour_magnitude,
                    e_lag=e_lag_magnitude
                )
                
                self._last_log_time = current_time

        # Set the initial state constraint for the OCP
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)



        ### ======================================= ###
        ### ============ Set parameters =========== ###
        ### ======================================= ###

        _,_,min_theta = self.compute_min_distance_to_trajectory(obs["pos"], self.theta)
        ## Debugging print for theta
        # print(f"theta: {self.theta:.2f}")
        self.pos = obs["pos"]

        # update trajectory
        self._handle_unified_update(obs)


        ### Try to penalisse distance to obstacles which did not work ###
        # # to correct: Unused ???
        # obs_array = obs["obstacles_pos"]          # shape=(4,3)
        # obs_flat = obs_array.reshape(-1)

        # Adjust progress speed based on trajectory curvature
        # Slower progress in high-curvature sections for better tracking
        kappa = self.curvature(self.theta)
        alpha = 0.12  # Curvature influence factor
        self.v_theta = self.base_v_theta / (1 + alpha * kappa)

        # currently mainly used for visulization
        stage_pos    = []
        nearest_pos  = []
        self._stage_pos   = np.asarray(stage_pos)
        self._nearest_pos = np.asarray(nearest_pos)

        # print(self.theta)

        for j in range(self.N + 1):
            # Compute the theta value for the current step
            theta_j = min(self.theta + j * self.v_theta * self.dt, 1.0)
            theta_j_next = min(theta_j + 0.0001, 1.0)

            weight_for_curr_theta = self.weight_for_theta(theta_j)

            p_ref = np.array(
                [
                    self.cs_x(theta_j),
                    self.cs_y(theta_j),
                    self.cs_z(theta_j),
                    self.cs_x(theta_j_next),
                    self.cs_y(theta_j_next),
                    self.cs_z(theta_j_next),
                    weight_for_curr_theta,
                ]
            )
            self.acados_ocp_solver.set(j, "p", p_ref)




        ### ======================================= ###
        ### ============ Solve the OCP ============ ###
        ### ======================================= ###
        self.acados_ocp_solver.solve()
        #This line retrieves the predicted state vector at stage 1 of the MPC horizon and stores it in x1.
        x1 = self.acados_ocp_solver.get(1, "x")

        # print(f"Total cost: {J:.3f}")

        ### ======================================= ###
        ### ============ Debugging ================ ###
        ### ======================================= ###

        # print_output(tick=self._tick, obs=obs, freq=self.config.env.freq)

        # u1 = self.acados_ocp_solver.get(1, "u")
        # # Debugging of  feedback law i.e print u1
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
        print(f"Time: {self._tick/self.freq:.2f}s")
        # print the current drone positions only
        print(f"Tick {self._tick}: px={x1[0]:.4f}, py={x1[1]:.4f}, pz={x1[2]:.4f}")

        # --- Update State for Next Iteration ---
        w = 1 / self.config.env.freq / self.dt
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]

        # Update trajectory progress
        self.theta = min(1.0, self.theta + self.v_theta * self.dt)

        # Extract and return the control command
        cmd = x1[10:14].copy()
        # cmd[0] *= 0.75 #für 0.25
        cmd[0] *= 1 

        return cmd




    def step_callback(self, action: NDArray[np.floating], obs: dict[str, NDArray[np.floating]], 
                    reward: float, terminated: bool, truncated: bool, info: dict) -> bool:
        """Callback after each environment step.
        
        Args:
            action: The control action applied in the step.
            obs: The observation after the step.
            reward: The reward received.
            terminated: Whether the episode has terminated.
            truncated: Whether the episode was truncated.
            info: Additional information from the environment.

        Returns:
            A boolean indicating if the callback was successful.
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

    def episode_callback(self, curr_time: float = None):
        """Callback function executed at the end of an episode.

        Args:
            curr_time: The final time of the episode.
        """
        if self.logger:
            self.logger.log_final_positions(
                gates_pos=self._info.get("gates_pos"),
                obstacles_pos=self._info.get("obstacles_pos")
            )
            self.logger.close()
        # -------------- Plot erzeugen ---------------------------------
        try:
            plot_script = pathlib.Path("plots/plot_speed.py").resolve()   # Pfad anpassen, wenn nötig
            subprocess.run(
                [sys.executable, str(plot_script), str(self.logger.run_dir)],
                check=True
            )
        except Exception as e:
            print(f"[WARN] Speed-Plot konnte nicht erzeugt werden: {e}")







    def get_trajectory(self) -> NDArray[np.floating]:
        return self._trajectory_history
    
    def get_waypoints(self) -> NDArray[np.floating]:
        return {
            'waypoints': self.waypoints,
            'gate_indices': self.gate_indices,
            'waypoint_blocks': self.waypoint_blocks,
            'gate_to_waypoint_mapping': self.gate_to_waypoint_mapping
        }

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


    def compute_min_distance_to_trajectory(self, drone_pos: np.ndarray,
                                        theta_hint: float | None = None):
        """
        Liefert den minimalen Abstand von `drone_pos` zur Spline-Trajektorie
        in einem kleinen Fenster um `theta_hint`.

        Args
        ----
        drone_pos   : np.ndarray, shape (3,)
        theta_hint  : geschätztes Bahn-θ, um das gesucht wird.
                    • None  → verwende self.theta  (Ist-Position)
                    • sonst → Center des Suchintervalls

        Returns
        -------
        min_dist     : float              – minimaler Abstand
        min_traj_pos : np.ndarray, (3,)   – Punkt p* auf der Trajektorie
        min_theta    : float              – Parameter θ*, zu dem p* gehört
        """
        # --- 0) Zentrum des Fensters festlegen ------------------------------
        if theta_hint is None:
            theta_hint = self.theta

        # --- 1) Parameter --------------------------------------------
        half_width     = 0.1      # ±-Fenster (0.05 ≙ 5 % der Spline­länge)
        coarse_samples = 5
        fine_samples   = 10

        search_start = max(0.0, theta_hint - half_width)
        search_end   = min(1.0, theta_hint + half_width)

        # --- 2) Grobe Suche ------------------------------------------
        min_dist   = np.inf
        min_theta  = theta_hint
        min_traj_pos = None

        for θ in np.linspace(search_start, search_end, coarse_samples):
            p = np.array([self.cs_x(θ), self.cs_y(θ), self.cs_z(θ)])
            d = np.linalg.norm(drone_pos - p)
            if d < min_dist:
                min_dist, min_theta, min_traj_pos = d, θ, p

        # --- 3) Feine Suche um das aktuelle Minimum ------------------
        δ = half_width / coarse_samples
        fine_start = max(0.0, min_theta - δ)
        fine_end   = min(1.0, min_theta + δ)

        for θ in np.linspace(fine_start, fine_end, fine_samples):
            p = np.array([self.cs_x(θ), self.cs_y(θ), self.cs_z(θ)])
            d = np.linalg.norm(drone_pos - p)
            if d < min_dist:
                min_dist, min_theta, min_traj_pos = d, θ, p

        return min_dist, min_traj_pos, min_theta



    
    def get_visualization_data(self, drone_pos):
        """Calculate visualization data for plotting error vectors and reference points.
        
        Args:
            drone_pos: The current position of the drone.
            theta_hint: An optional hint for the search start point on the trajectory.

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
        trajectory_updated = False  # Initialize this variable
        
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

    # currently mainly used for visulization
    def get_stage_vs_nearest(self):
        """Returns stored stage and nearest point data for analysis."""
        return self._stage_pos, self._nearest_pos

    def weight_for_theta(self, theta: float) -> float:
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
        # Loop over all gates and their associated peak weights
        for gate_theta, peak_weight in zip(self.gate_thetas, self.gate_peak_weights):
            # Compute the normalized distance from the current theta to the gate's theta
            diff = (theta - gate_theta) / self.sigma  # self.sigma controls the width of the peak
            # Compute the Gaussian-shaped influence of this gate
            gate_influence = (peak_weight - self.base_weight) * np.exp(-0.5 * diff * diff)
            # The weight at this theta is the maximum of the base weight and all gate influences
            w = max(w, self.base_weight + gate_influence)
        # Return the final weight for this theta
        return w

