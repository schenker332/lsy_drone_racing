from __future__ import annotations 
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control import Controller
if TYPE_CHECKING:
    from numpy.typing import NDArray
from lsy_drone_racing.control.mpcc_curv_1_utils.mpcc_curv_1_ocp_solver import create_ocp_solver
from lsy_drone_racing.control.helper.print_output import print_output
from lsy_drone_racing.control.helper.datalogger import DataLogger



# Set peak weights for the Gaussian-like cost function around each gate]
# Set individual sigma values for each gate (controls the width of the Gaussian peak); 0.01 5% is of the norm scale
GATE_WEIGHT_CONFIG = {
    0: {"peak_weight": 500, "sigma": 0.02},   # Gate 0: narrow peak
    1: {"peak_weight": 300, "sigma": 0.02}, # Gate 1: wider peak  
    2: {"peak_weight": 300, "sigma": 0.02},  # Gate 2: narrow peak
    3: {"peak_weight": 300, "sigma": 0.02}  # Gate 3: very narrow, high peak
}


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
        # Hol dir die beiden Werte
        mcfg = config.mpc

        
        ### ====================================================================== ###
        ### =========================== Waypoints ================================ ###
        ### ====================================================================== ###

        # Gate-Positionen
        gates = np.array([
            [0.45, -0.50, 0.56],  # Gate 0      [0.45, -0.5, 0.56]
            [1.0, -1.05, 1.25],   # Gate 1      [1.0, -1.05, 1.11]
            [0.0, 1.0, 0.56],     # Gate 2      [0.0, 1.0, 0.56]
            [-0.5, 0.0, 1.11]     # Gate 3      [-0.5, 0.0, 1.11]
        ])
        
        # Start bis Gate 0
        b0 = np.array([
            [1.0, 1.5, 0.2],
            [0.8, 1.0, 0.2],
            [0.7, 0.1, 0.4]
        ])
        
        # Gate 0 -- Gate 1
        b1 = np.array([
            [0.2, -0.7, 0.65],  
            [0.1, -1.0, 0.75],
            [0.5, -1.5, 0.8]
        ])
        
        # Gate 1 -- Gate 2
        b2 = np.array([
            [1.15, -0.75, 1],
            [0.5, 0, 0.8]
        ])
        
        # Gate 2 -- Gate 3
        b3 = np.array([
            [-0.2, 1.5, 0.56],
            [-0.9, 1.3, 0.8],
            [-0.75, 0.5, 1.11]   
        ])
        
        # nach Gate 3 
        b4 = np.array([
            [-0.1, -1, 1.11]
        ])
        
        # Kombiniere alle Waypoints: b0 + gate0 + b1 + gate1 + b2 + gate2 + b3 + gate3 + b4
        waypoint_sections = [b0, gates[0:1], b1, gates[1:2], b2, gates[2:3], b3, gates[3:4], b4]
        waypoints = np.vstack(waypoint_sections)
        

        self.waypoints = waypoints

        
        ts = np.linspace(0, 1, np.shape(waypoints)[0])
        self.cs_x = CubicSpline(ts, waypoints[:, 0])
        self.cs_y = CubicSpline(ts, waypoints[:, 1])
        self.cs_z = CubicSpline(ts, waypoints[:, 2])


        ### ====================================================================== ###
        ### =========================== Mapping ================================== ###
        ### ====================================================================== ###


        # Automatische Indizes berechnen
        current_index = 0
        gate_indices = []
        waypoint_blocks = {}
        
        for i, section in enumerate(waypoint_sections):
            section_length = len(section)
            
            if i % 2 == 1:  # Ungerade Indizes sind Gates (1, 3, 5, 7)
                gate_indices.append(current_index)
            else:  # Gerade Indizes sind Waypoint-Blöcke (0, 2, 4, 6, 8)
                block_idx = i // 2  # 0, 1, 2, 3, 4
                waypoint_blocks[block_idx] = list(range(current_index, current_index + section_length))
            
            current_index += section_length
        
        # Store strukturierte Daten
        self.waypoint_blocks = waypoint_blocks
        self.gate_indices = gate_indices

        
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

            "o2": [  # Wenn sich Obstacle 2 ändert  
            ("b3.0", 1.0, 1.0, 0.0, [0.0, 0.0, 0.0]),  # Block 2, Waypoint 1
            ],

            "o3": [  # Wenn sich Obstacle 3 ändert
            ("b3.2", 1.5, 1.5, 1.5, [0.0, 0.0, 0.0]),  # Block 3, Waypoint 2 (at obstacle 4)
            ],
        }
        

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

        ### ======================== Logger ======================== ###
        self.logging_enabled = True
        if  self.logging_enabled:
            self.logger = DataLogger()
            self._last_log_time = -1
        else:
            self.logger = None
        ### ======================================================== ###

        # MPC parameters
        self.N = mcfg.N                   # Number of discretization steps
        self.T_HORIZON = mcfg.T_HORIZON   # Time horizon in seconds
        self.dt = self.T_HORIZON / self.N  # Step size

        # Initialize state variables    
        self.last_f_collective = 0.3  # Current thrust value
        self.last_f_cmd = 0.3     # Commanded thrust value
        self.last_rpy_cmd = np.zeros(3)  # Commanded roll, pitch, yaw

        # Store configuration and tracking variables
        self.config = config
        self.finished = False
        self._info = info

        self._last_gates_visited = None  # Initialize gate tracking
        self._last_gates_pos = None      # Store previous gate positions for delta calculation
        self._last_obstacles_pos = None  # Store previous obstacle positions



        self.theta = 0
        t= mcfg.t_scaling
        self.v_theta = 1/ (t * self.dt * self.freq) 

        dx   = self.cs_x.derivative(1)
        ddx  = self.cs_x.derivative(2)
        dy   = self.cs_y.derivative(1)
        ddy  = self.cs_y.derivative(2)
        dz   = self.cs_z.derivative(1)
        ddz  = self.cs_z.derivative(2)

        def curvature(theta):
            v = np.array([dx(theta), dy(theta), dz(theta)])
            a = np.array([ddx(theta), ddy(theta), ddz(theta)])
            num = np.linalg.norm(np.cross(v, a))
            den = np.linalg.norm(v)**3 + 1e-8
            return num/den
        

        self.curvature = curvature
        self.alpha_curv_speed = mcfg.alpha_curv_speed    
        self.base_v_theta = self.v_theta

        # Automatische Gate-Thetas basierend auf berechneten Indizes
        self.gate_thetas = [ts[i] for i in gate_indices]

         # Extract peax wewights and peak sigmas from the configuration
        # self.gate_peak_weights = [GATE_WEIGHT_CONFIG[i]["peak_weight"] for i in range(4)]
        # self.gate_sigmas = [GATE_WEIGHT_CONFIG[i]["sigma"] for i in range(4)]
        self.gate_peak_weights = [mcfg.peak_weight, mcfg.peak_weight, mcfg.peak_weight, mcfg.peak_weight] 
        self.gate_sigmas = [mcfg.sigma, mcfg.sigma, mcfg.sigma, mcfg.sigma]

        self.base_weight = mcfg.base_weight 

        # Create the optimal control problem solver
        self.acados_ocp_solver, self.ocp = create_ocp_solver(
            self.T_HORIZON,
            self.N,
            config.mpc,       # ← übergib hier dein ConfigDict mit max_v_theta etc.
            verbose=False
        )



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
        xcurrent = np.concatenate((obs["pos"], obs["vel"],  R.from_quat(obs["quat"]).as_euler("xyz", degrees=False), [self.last_f_collective, self.last_f_cmd], self.last_rpy_cmd) )
        

        ### ======================== Logger ======================== ###
        # Log state vector every 0.1 seconds
        if self.logger:
            current_time = self._tick / self.freq
            # nur alle 0.01 s loggen
            if current_time - self._last_log_time >= 0.01:
                ref_pt = np.array([self.cs_x(self.theta), self.cs_y(self.theta), self.cs_z(self.theta)])
                u1 = self.acados_ocp_solver.get(1, "u")


                kappa = self.curvature(self.theta)
                self.logger.log_state(current_time, xcurrent, u1, ref_point=ref_pt, curvature=kappa, v_theta=self.v_theta)
                self._last_log_time = current_time


                # current_weight = self.weight_for_theta(self.theta)
                # vis_data = self.get_contour_lag_error(obs["pos"])
                # e_contour_magnitude = np.linalg.norm(vis_data['e_c_vec'])
                # e_lag_magnitude = abs(vis_data['e_l_scalar'])
                
                # # Log weight data with actual error values    
                # self.logger.log_weight_data(
                #     current_time, 
                #     current_weight,
                #     e_contour=e_contour_magnitude,
                #     e_lag=e_lag_magnitude
                # )

        ### ======================================================== ###
                
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)

        ### ======================================= ###
        ### ============ Set parameters =========== ###
        ### ======================================= ###

        # update trajectory
        self._handle_unified_update(obs)

        κ = self.curvature(self.theta)
        
        self.v_theta = self.base_v_theta / (1 + self.alpha_curv_speed * κ)




        for j in range(self.N + 1):
            theta_j = min(self.theta + j * self.v_theta * self.dt, 1.0)
            theta_j_next = min(theta_j + 0.0001, 1.0)

            p_ref = np.array([
                self.cs_x(theta_j), self.cs_y(theta_j), self.cs_z(theta_j),  # for contouring
                self.cs_x(theta_j_next), self.cs_y(theta_j_next), self.cs_z(theta_j_next),  # for contouring
                self.weight_for_theta(theta_j),  # for gauss weighting
            ])

            self.acados_ocp_solver.set(j, "p", p_ref)


        ### ======================================= ###
        ### ============ Solve the OCP ============ ###
        ### ======================================= ###
        self.acados_ocp_solver.solve()
        x1 = self.acados_ocp_solver.get(1, "x")


        ### ======================================= ###
        ### ============ Update state ============= ###
        ### ======================================= ###

        w = 1 / self.config.env.freq / self.dt
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]  

        self.theta = min(1.0, self.theta + self.v_theta * self.dt)
        cmd = x1[10:14].copy()
        # cmd[0] *= 0.75 #für 0.25
        cmd[0] *= 1 

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
        return self.finished

    def episode_reset(self):
        """Reset controller state for a new episode."""
        self._tick = 0  # Important for timing and next episode


    def episode_callback(self, curr_time: float=None):
        """Callback at the end of each episode.
        
        Args:
            curr_time: Current simulation time.
        """
        pass


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

    def get_contour_lag_error(self, drone_pos):
        """Calculate visualization data for plotting error vectors and reference points.
        
        Args:
            drone_pos: Current drone position as np.array([x, y, z])
            
        Returns:
            dict: Dictionary containing all visualization data with keys:
                - ref_point: Reference point on trajectory
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
            't_hat_scaled': t_hat_scaled,
            'e_l_vis': e_l_vis,
            'e_c_vis': e_c_vis,
            'e_c_vec': e_c_vec,
            'e_l_scalar': e_l_scalar

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