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

# Add this after the GATE_ORTHOGONAL_WP_DISTANCE constant
GATE_ORTHOGONAL_CONFIG = {
    0: {"before": 0.05, "after": 0.3, "center": False},     # Gate 0: before + after
    1: {"before": 0.3, "after": 0.25, "center": False},     # Gate 1: before + after
    2: {"before": None, "after": None, "center": True},     # Gate 2: only center waypoint
    3: {"before": 0.05, "after": None, "center": False}       # Gate 3: before + after
}


# Set peak weights for the Gaussian-like cost function around each gate]
# Set individual sigma values for each gate (controls the width of the Gaussian peak); 0.01 5% is of the norm scale
GATE_WEIGHT_CONFIG = {
    0: {"peak_weight": 1500, "sigma": 0.04},   # Gate 0: narrow peak
    1: {"peak_weight": 300, "sigma": 0.02}, # Gate 1: wider peak  
    2: {"peak_weight": 1500, "sigma": 0.08},  # Gate 2: narrow peak
    3: {"peak_weight": 300, "sigma": 0.04}  # Gate 3: very narrow, high peak
}

V_THETA_MAX = 0.13  # Base rate of progress, can be adjusted dynamically

def orthogonal_waypoints(gates_obs: np.ndarray,
                        gates_quat: np.ndarray = None,
                        dist: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute waypoints before and after a gate along its approach direction.
    gate_quat is assumed in [x, y, z, w] order.
    """
    if gates_quat is not None:
        # Use gate orientation - Y-axis is the approach direction
        rot = R.from_quat(gates_quat)
        normal = rot.apply(np.array([0.0, 1.0, 0.0]))  # Changed from [1,0,0] to [0,1,0]
    else:
        # Default: approach direction (forward-backward in flight direction)
        normal = np.array([0.0, 1.0, 0.0])  # Changed from [1,0,0] to [0,1,0]
    
    # normalize to unit length
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    before = gates_obs - dist * normal  # Point before the gate
    after = gates_obs + dist * normal   # Point after the gate
    return before, after


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
        # y is towards the wall and x towards the comp
        # ])

        ### Original Gate-Positions
        gates = np.array([
            [0.45, -0.5, 0.56],     # Gate 0      
            [1.0, -1.05, 1.11],     # Gate 1     
            [0.0, 1.0, 0.56],       # Gate 2      
            [-0.5, 0.0, 1.2]       # Gate 3     
        ]) 

        # Generate orthogonal waypoints:
        gate_orthogonal_waypoints = []

        for gate_idx in range(4):
            gate_wp_config = GATE_ORTHOGONAL_CONFIG[gate_idx]
            gate_pos = obs["gates_pos"][gate_idx]
            gate_quat = obs["gates_quat"][gate_idx]
            
            before_wp = None
            after_wp = None
            
            if gate_wp_config["before"] is not None:
                before_wp, _ = orthogonal_waypoints(gate_pos, gate_quat, dist=gate_wp_config["before"])
            
            if gate_wp_config["after"] is not None:
                _, after_wp = orthogonal_waypoints(gate_pos, gate_quat, dist=gate_wp_config["after"])
            
            gate_orthogonal_waypoints.append((before_wp, after_wp))

        # Start bis Gate 0
        b0 = np.array([
            [1.0, 1.5, 0.2],
            [0.95, 1.0, 0.3],
            [0.7, 0.1, 0.4],
        ])

        # Gate 0: before + after waypoints (no center)
        g0_before, g0_after = gate_orthogonal_waypoints[0]
        g0_section = np.array([g0_before, g0_after])

        # Gate 0 -- Gate 1
        b1 = np.array([
            [0.2, -1.4, 0.85],
        ])

        # Gate 1: before + after waypoints (no center)
        g1_before, g1_after = gate_orthogonal_waypoints[1]
        g1_section = np.array([g1_before, g1_after])

        # Gate 1 -- Gate 2 (removed [1.15, -0.75, 1] as requested)
        b2 = np.array([
            [0.65, -0.25, 0.85],
        ])

        # Gate 2: only center waypoint
        g2_section = np.array([obs["gates_pos"][2]])

        # Gate 2 -- Gate 3
        b3_1 = np.array([
            [-0.2, 1.5, 0.56],
            [-0.9, 1.3, 0.8],   
        ])

        b3_2 = np.array([
            [-0.75, 0.5, 1.11],     
        ])

        # Gate 3: before + after waypoints (no center)
        g3_before, g3_after = gate_orthogonal_waypoints[3]
        # g3_section = np.array([g3_before, g3_after])
        g3_section = np.array([g3_before])

        # nach Gate 3 
        b4 = np.array([
            [-0.5, -2, 1.11],
            [-0.5, -6, 1.11],
        ])

        # Combine all waypoints
        waypoint_sections = [
        ("block", "b0", b0),
        ("gate", "g0", g0_section),
        ("block", "b1", b1),
        ("gate", "g1", g1_section),
        ("block", "b2", b2),
        ("gate", "g2", g2_section),
        ("block", "b3_1", b3_1),
        ("block", "b3_2", b3_2),
        ("gate", "g3", g3_section),
        ("block", "b4", b4)
    ]
        # Stack all waypoints for the spline
        waypoints = np.vstack([section[2] for section in waypoint_sections])

        # Calculate indices with explicit section handling
        current_index = 0
        gate_indices = []
        orthogonal_indices = {}
        waypoint_blocks = {}
        gate_counter = 0
        block_counter = 0

        for section_type, section_name, section_data in waypoint_sections:
            section_length = len(section_data)
            
            if section_type == "gate":
                gate_idx = gate_counter
                gate_wp_config = GATE_ORTHOGONAL_CONFIG[gate_idx]
                
                if gate_wp_config["center"]:  # Gate 2: only center waypoint
                    gate_indices.append(current_index)
                    
                else:  # Gates 0, 1, 3: before + after waypoints
                    orthogonal_indices[gate_idx] = {
                        'before': current_index,
                        'after': current_index + 1 if section_length > 1 else None
                    }
                    # Use middle index for approximation
                    gate_indices.append(current_index + 0.5 if section_length > 1 else current_index)
                    
                gate_counter += 1
                
            elif section_type == "block":
                waypoint_blocks[section_name] = list(range(current_index, current_index + section_length))
                block_counter += 1
            
            current_index += section_length

        # Create mapping only for gates with actual integer indices
        self.gate_to_waypoint_mapping = {}
        for i, gate_idx in enumerate(gate_indices):
            if isinstance(gate_idx, int):  # Only integer indices
                self.gate_to_waypoint_mapping[i] = gate_idx

        # Store strukturierte Daten
        self.waypoint_blocks = waypoint_blocks
        self.gate_indices = gate_indices
        self.orthogonal_indices = orthogonal_indices  # Store the orthogonal waypoint indices
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
            "g0": [  # Gate 0: before + after waypoints
                ("g0_before", 1.05, 1.0, 1.0, [0.0, 0.0, 0.0]),
                ("g0_after", 1.05, 1.0, 1.0, [0.0, 0.0, 0.0]),
            ],

            "g1": [  # Gate 1: before + after waypoints
                ("g1_before", 1.0, 1.0, 1.0, [0.0, 0.0, 0.0]),
                ("g1_after", 1.0, 1.0, 1.0, [0.0, 0.0, 0.0]),
            ],

            "g2": [  # Gate 2: only center waypoint
                ("g2", 1.0, 1.0, 1.0, [0.0, 0.0, 0.0]),
                ("b2.0", 0.3, 0.5, 0.0, [0.0, 0.0, 0.0]),
            ],

            "g3": [  # Gate 3: before waypoint only
                ("g3_before", 1.0, 1.0, 1.0, [0.0, 0.0, 0.0]),
                #("g3_after", 1.0, 1.0, 1.0, [0.0, 0.0, 0.0]),
            ],

            "o1": [
                ("b1.0", 1.0, 1.0, 0.0, [0.0, 0.0, 0.0]),
            ],

            "o2": [  # Obstacle 2 mapping
                ("b3_1.0", 1.0, 1.0, 0.0, [0.0, 0.0, 0.0]),
                ("g2", 0.5, 0.5, 1.0, [0.0, 0.0, 0.0]),
            ],

            "o3": [
                ("b3_2.0", 2.0, 2.0, 0.0, [0.0, 0.0, 0.0]),
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
        t = 5.5
        self.v_theta = max(1 / (t * self.dt * self.freq), V_THETA_MAX)  # Base rate of progress
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
        # Find the `theta` values corresponding to each gate CENTER (not the waypoint indices)
        # For gates without center waypoints, calculate the theta at the gate position
        self.gate_thetas = []
        for gate_idx in range(4):
            gate_wp_config = GATE_ORTHOGONAL_CONFIG[gate_idx]
            
            if gate_wp_config["center"]:  # Gate has center waypoint
                # Use the waypoint index directly
                gate_wp_idx = gate_indices[gate_idx]
                self.gate_thetas.append(ts[gate_wp_idx])
            else:
                # Gate doesn't have center waypoint, approximate from orthogonal waypoints
                if gate_idx in self.orthogonal_indices:
                    # Get before and after indices
                    before_idx = self.orthogonal_indices[gate_idx].get('before', None)
                    after_idx = self.orthogonal_indices[gate_idx].get('after', None)
                    
                    if before_idx is not None and after_idx is not None:
                        # Both before and after exist, take average
                        theta_center = (ts[before_idx] + ts[after_idx]) / 2.0
                    elif before_idx is not None:
                        # Only before exists, estimate center using distance
                        before_dist = gate_wp_config["before"]
                        # Approximate theta step per unit distance
                        theta_step = before_dist / np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))
                        theta_center = ts[before_idx] + theta_step
                    elif after_idx is not None:
                        # Only after exists, estimate center using distance
                        after_dist = gate_wp_config["after"]
                        # Approximate theta step per unit distance
                        theta_step = after_dist / np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))
                        theta_center = ts[after_idx] - theta_step

                    
                self.gate_thetas.append(theta_center)

         # Extract peax wewights and peak sigmas from the configuration
        self.gate_peak_weights = [GATE_WEIGHT_CONFIG[i]["peak_weight"] for i in range(4)]
        self.gate_sigmas = [GATE_WEIGHT_CONFIG[i]["sigma"] for i in range(4)]

        # --- OCP Solver Setup ---
        self.acados_ocp_solver, self.ocp = create_ocp_solver(self.T_HORIZON, self.N)

        # --- Contouring Cost Weighting ---
        # Base weight for the contouring error cost
        self.base_weight = 130.0
        # Sigma for the Gaussian-like weight distribution around gates

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
                self.last_rpy_cmd
            )
        )

        # Log state vector periodically
        if self.logger:
            current_time = self._tick / self.freq
            if current_time - self._last_log_time >= 0.01:
                
                ### Leftover abandoned test calculate and update the curvature for a future theta
                # calculate kappa over a larger horizon does not seem to make a big difference. tried for N 5-20
                # prediction_step = 10
                # theta_N10 = min(self.theta + prediction_step * self.v_theta * self.dt, 1.0)
                # kappaN0 = self.curvature(self.theta)
                # kappaN9 = self.curvature(theta_N10)
                # kappa = (kappaN0 + kappaN9) / 2.0
                 # print(f"Curv N=0: {kappaN0}, Curv N=9: {kappaN9}", "combined:", kappa)

                # Calculate current curvature and distance
                kappa = self.curvature(self.theta)

               

                
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
                        curvature=kappa
                    )
                except:
                    self.logger.log_state(
                        current_time, xcurrent, 
                        ref_point=ref_pt, 
                        curvature=kappa
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


        # update trajectory
        self._handle_unified_update(obs)



        kappa = self.curvature(self.theta)
        alpha = 0.12  # Curvature influence factor
        self.v_theta = self.base_v_theta / (1 + alpha * kappa)




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
        x1 = self.acados_ocp_solver.get(1, "x")


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
        trajectory_updated = False
        
        for target, x_factor, y_factor, z_factor, offset in self.response_mapping[trigger_id]:
            
            # Calculate response with factors and offset
            response_delta = np.array([
                delta[0] * x_factor + offset[0],
                delta[1] * y_factor + offset[1], 
                delta[2] * z_factor + offset[2]
            ])
            
            # Apply to target
            if target.startswith("g") and not "_" in target:  # Gate center target
                gate_idx = int(target[1:])
                if gate_idx in self.gate_to_waypoint_mapping:
                    wp_idx = self.gate_to_waypoint_mapping[gate_idx]
                    if 0 <= wp_idx < len(self.waypoints):
                        self.waypoints[wp_idx] += response_delta
                        trajectory_updated = True
                        
            elif target.startswith("g") and "_" in target:  # Orthogonal gate waypoints
                parts = target.split("_")
                gate_idx = int(parts[0][1:])
                direction = parts[1]  # "before" or "after"
                
                if gate_idx in self.orthogonal_indices:
                    if direction in self.orthogonal_indices[gate_idx]:
                        wp_idx = self.orthogonal_indices[gate_idx][direction]
                        if wp_idx is not None and 0 <= wp_idx < len(self.waypoints):
                            # Update orthogonal waypoint with new gate position + orthogonal offset
                            new_gate_pos = new_pos
                            gate_quat = self._info.get("gates_quat", [None, None, None, None])[gate_idx]
                            gate_wp_config = GATE_ORTHOGONAL_CONFIG[gate_idx]
                            
                            if direction == "before" and gate_wp_config["before"] is not None:
                                before, _ = orthogonal_waypoints(new_gate_pos, gate_quat, gate_wp_config["before"])
                                self.waypoints[wp_idx] = before
                            elif direction == "after" and gate_wp_config["after"] is not None:
                                _, after = orthogonal_waypoints(new_gate_pos, gate_quat, gate_wp_config["after"])
                                self.waypoints[wp_idx] = after
                            trajectory_updated = True
                            
            elif target.startswith("b"):  # Block waypoint target
                # Parse "b2.0" or "b3_1.0" -> section_name, waypoint_in_block
                parts = target.split(".")
                section_name = parts[0]
                wp_in_block = int(parts[1])
                
                if section_name in self.waypoint_blocks:
                    if wp_in_block < len(self.waypoint_blocks[section_name]):
                        wp_idx = self.waypoint_blocks[section_name][wp_in_block]
                        if 0 <= wp_idx < len(self.waypoints):
                            self.waypoints[wp_idx] += response_delta
                            trajectory_updated = True
        # if trigger_id == "o2":
        #     # Special handling for obstacle 2
        #     for target in ["b3_1.0", "b3_1.1"]:
        #         response_delta = self._calculate_obstacle2_offset(new_pos, old_pos, target)
                
        #         parts = target.split(".")
        #         section_name = parts[0]
        #         wp_in_block = int(parts[1])
                
        #         if section_name in self.waypoint_blocks:
        #             if wp_in_block < len(self.waypoint_blocks[section_name]):
        #                 wp_idx = self.waypoint_blocks[section_name][wp_in_block]
        #                 if 0 <= wp_idx < len(self.waypoints):
        #                     self.waypoints[wp_idx] += response_delta
        #             trajectory_updated = True

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

