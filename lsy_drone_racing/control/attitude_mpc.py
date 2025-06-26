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
from lsy_drone_racing.control.arc_length_parametrization import arc_length_parametrization



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
        self.N = 25                    # Number of discretization steps
        self.T_HORIZON = 1           # Time horizon in seconds
        self.dt = self.T_HORIZON / self.N  # Step size

        # Track waypoints with gates labeled
        waypoints = np.array(
        [
            [1.0, 1.5, 0.3],
            [0.8, 1.0, 0.2],
            [0.7, 0.1, 0.4],
            [0.45, -0.5, 0.56],  # gate1
            [0.2, -0.7, 0.65],
            [0.5, -1.5, 0.8],
            [1, -1.05, 1.11],    # gate2
            [1.15, -0.75, 1],
            [0.5, 0, 0.8],
            [0, 1, 0.56],        # gate3
            [-0.1, 1.2, 0.56],
            [-0.3, 1.2, 1.1],
            [-0.2, 0.4, 1.1],
            [-0.45, 0.1, 1.11],
            [-0.5, 0, 1.11],     # gate4
            [-0.5, -0.2, 1.11],
        ])
        
        
        # ======================== old trajectory ==================== ###
        # ts = np.linspace(0, 1, np.shape(waypoints)[0])
        # cs_x = CubicSpline(ts, waypoints[:, 0])
        # cs_y = CubicSpline(ts, waypoints[:, 1])
        # cs_z = CubicSpline(ts, waypoints[:, 2])

        # # Generate time-based reference trajectory
        # self._des_completion_time =  
        # ts = np.linspace(0, 1, int(self.freq * self._des_completion_time))
        # self.x_des = cs_x(ts)
        # self.y_des = cs_y(ts)
        # self.z_des = cs_z(ts)
        # ========================= old trajectory ==================== ###



        # ### ==================== new trajectory ==================== ###


        # --- VERSION 1: Current Arc-Length Parametrization (Comment out to test Version 2) ---
        theta_k_values, x_vals, y_vals, z_vals, _, _, _,_,_ = arc_length_parametrization(waypoints, num_samples=1000)

        # --- VERSION 2: Simpler Time-Based Parametrization similar to old one (Uncomment to test) ---
        # # This ignores true distance and can create a smoother, but less accurate, path.
        # # It's a good diagnostic tool.
        # num_waypoints = waypoints.shape[0]
        # t_values = np.linspace(0, 1, num_waypoints) # Dummy "time" parameter from 0 to 1
        # x_vals = waypoints[:, 0]
        # y_vals = waypoints[:, 1]
        # z_vals = waypoints[:, 2]
        # # rename t_values to theta_k_values to avoid changing the rest of the code.
        # theta_k_values = t_values



        ### new ###  Store path for progress estimation
        self.theta_grid = theta_k_values
        self.path_points = np.vstack((x_vals, y_vals, z_vals)).T

        self.cs_x = CubicSpline(theta_k_values, x_vals)
        self.cs_y = CubicSpline(theta_k_values, y_vals)
        self.cs_z = CubicSpline(theta_k_values, z_vals)

        self.theta = 0.0        # Fortschrittsstartwert
        # self.v_theta = 0.1      # Fortschrittsgeschwindigkeit 0.66 theta/s

        # Initialize progress state variables
        self.last_theta = 0.0
        self.last_vtheta = 0.0

        # self.time = 8
        # self.v_theta = 1 / (self.time * self.dt * self.freq)

        ### ==================================================================== ###




       

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
        # # Get the current reference point index based on timing
        # i = min(self._tick, len(self.x_des) - 1)
        # if self._tick > i:
        #     self.finished = True

        if self.theta >= 1.2:
            self.finished = True
            
        print_output(tick=self._tick, obs=obs, freq=self.config.env.freq)
        # Convert quaternion to roll-pitch-yaw angles
        rpy = R.from_quat(obs["quat"]).as_euler("xyz", degrees=False)

        ### new --- Project current position onto path to get current_theta ---
        dists = np.linalg.norm(self.path_points - obs["pos"], axis=1)
        current_theta = self.theta_grid[np.argmin(dists)]

        # Construct the current state vector for the MPC solver
        xcurrent = np.concatenate((obs["pos"], obs["vel"], rpy, [self.last_f_collective, self.last_f_cmd], self.last_rpy_cmd, [current_theta, self.last_vtheta] ))
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)



        ### ====================== old trajectory + old cost function ==================== ###
        # # Set reference trajectory for each step in the prediction horizon
        # for j in range(self.N):
        #     idx = min(i + j, len(self.x_des) - 1)
        #     yref = np.array([self.x_des[idx], self.y_des[idx], self.z_des[idx], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        #     self.acados_ocp_solver.set(j, "yref", yref)

        # # Set terminal state reference
        # idx_N = min(i + self.N, len(self.x_des) - 1)
        # yref_N = np.array([self.x_des[idx_N], self.y_des[idx_N], self.z_des[idx_N], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0])
        # self.acados_ocp_solver.set(self.N, "yref", yref_N)
        # ### ================================================================ ###



        ## ==================== new trajectory + old cost function ==================== ###

        # for j in range(self.N):
        #     theta_j = min(self.theta + j * self.v_theta * self.dt, 1.0)

        #     xj = self.cs_x(theta_j)
        #     yj = self.cs_y(theta_j)
        #     zj = self.cs_z(theta_j)

        #     yref = np.array([
        #         xj, yj, zj, 
        #         0.0, 0.0, 0.0,  # vels
        #         0.0, 0.0, 0.0,  # rpy
        #         0.35, 0.35,     # collective + f_cmd
        #         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # rest
        #     ])
        #     self.acados_ocp_solver.set(j, "yref", yref)



        # theta_N = min(self.theta + self.N * self.v_theta * self.dt, 1.0)
        # xN = self.cs_x(theta_N)
        # yN = self.cs_y(theta_N)
        # zN = self.cs_z(theta_N)

        # yref_N = np.array([xN, yN, zN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0])
        # self.acados_ocp_solver.set(self.N, "yref", yref_N)

        ## ================================================================== ###



        ### =================== old trajectory + new cost function ==================== ###
        # for j in range(self.N):
        #     idx = min(i + j, len(self.x_des) - 1)
        #     xj = self.x_des[idx]
        #     yj = self.y_des[idx]
        #     zj = self.z_des[idx]

        #     idx_next = min(idx + 1, len(self.x_des) - 1)
        #     xj_next = self.x_des[idx_next]
        #     yj_next = self.y_des[idx_next]
        #     zj_next = self.z_des[idx_next]

        #     p_ref = np.array([xj, yj, zj, xj_next, yj_next, zj_next])
        #     self.acados_ocp_solver.set(j, "p", p_ref)


        # # Set terminal state reference
        # idx_N = min(i + self.N, len(self.x_des) - 1)

        # xN = self.x_des[idx_N]
        # yN = self.y_des[idx_N]
        # zN = self.z_des[idx_N]

        # idx_N_next = min(idx_N + 1, len(self.x_des) - 1)
        # xN_next = self.x_des[idx_N_next]
        # yN_next = self.y_des[idx_N_next]
        # zN_next = self.z_des[idx_N_next]

        # p_ref_N = np.array([xN, yN, zN, xN_next, yN_next, zN_next])
        # self.acados_ocp_solver.set(self.N, "p", p_ref_N)
        # ### ================================================================== ###



        ### =================== new trajectory + new cost function ==================== ###

        ### Predict theta for the horizon based on the last optimal velocity
        # this will produce the following ouput: first this: [0, v·dt, 2v·dt, …, (N+1)v·dt].
        # then this with the addition of thetha: [θ₀, θ₀ + v·dt, θ₀ + 2v·dt, …]

        # # old too much corner cutting
        # theta_prediction = current_theta + self.last_vtheta * np.arange(self.N + 2) * self.dt 



        # New.. to create a smooth and stable reference for the current step,
        # always re-anchoring the plan to the drone's actual position.
        try:
            # 1. Extract the optimal velocity profile from the previous solution.
            vtheta_profile = np.zeros(self.N)
            for j in range(self.N):
                x_pred = self.acados_ocp_solver.get(j + 1, "x") # Get plan for steps 1...N
                vtheta_profile[j] = x_pred[15]

            # 2. Integrate this velocity profile to get the new reference path.
            # We need N+2 points for the loop setting parameters up to the terminal stage.
            theta_prediction = np.zeros(self.N + 2)
            theta_prediction[0] = current_theta  # CRITICAL: Anchor plan to reality.

            for j in range(self.N + 1):
                # Use the velocity from the profile, or the last one if we're at the end.
                v_step = vtheta_profile[j] if j < self.N else vtheta_profile[-1]
                theta_prediction[j+1] = theta_prediction[j] + v_step * self.dt

        except Exception:
            # Fallback for the very first run when no previous solution exists.
            # Predict using a simple constant velocity, anchored to the current position.
            theta_prediction = current_theta + self.last_vtheta * np.arange(self.N + 2) * self.dt


        for j in range(self.N):
            # ### old version ###
            # theta_j = min(self.theta + j * self.v_theta * self.dt, 1.0)
            # theta_j_next = min(self.theta + (j + 1) * self.v_theta * self.dt, 1.0)

            ### new version ###
            theta_j = min(theta_prediction[j], 1.0)
            theta_j_next = min(theta_prediction[j + 1], 1.0)

            xj = self.cs_x(theta_j)
            yj = self.cs_y(theta_j)
            zj = self.cs_z(theta_j)

            xj_next = self.cs_x(theta_j_next)
            yj_next = self.cs_y(theta_j_next)
            zj_next = self.cs_z(theta_j_next)

            p_ref = np.array([xj, yj, zj, xj_next, yj_next, zj_next])
            self.acados_ocp_solver.set(j, "p", p_ref)



        # ### old
        # theta_N = min(self.theta + self.N * self.v_theta * self.dt, 1.0)
        # theta_N_plus = min(self.theta + (self.N + 1) * self.v_theta * self.dt, 1.0)
        theta_N = min(theta_prediction[self.N], 1.0)
        theta_N_plus = min(theta_prediction[self.N+1], 1.0)


        xN = self.cs_x(theta_N)
        yN = self.cs_y(theta_N)
        zN = self.cs_z(theta_N)

        xN_next = self.cs_x(theta_N_plus)
        yN_next = self.cs_y(theta_N_plus)
        zN_next = self.cs_z(theta_N_plus)

        p_ref_N = np.array([xN, yN, zN, xN_next, yN_next, zN_next])
        self.acados_ocp_solver.set(self.N, "p", p_ref_N)
        # # ### ================================================================ ###






        ####============================================================================================
        # Store previous theta to track progress milestones
        if not hasattr(self, 'last_progress_milestone'):
            self.last_progress_milestone = 0.0


        ####============================================================================================
        ### New ### The theta value is now part of the optimization problem and does not need to be manually updated here.
        ####============================================================================================
        # # Update theta value
        # self.theta = min(self.theta + self.v_theta * self.dt, 1.0)






        # Solve the MPC problem
        self.acados_ocp_solver.solve()
        # total_cost = self.acados_ocp_solver.get_cost()
        # print(f"Solver Kosten: {total_cost:.4f}")
        x1 = self.acados_ocp_solver.get(1, "x")
        # Apply low-pass filter to thrust for smoother control
        w = 1 / self.config.env.freq / self.dt
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]
        ### new  for MPCC
        self.last_theta = x1[14] # Update from solver result
        self.last_vtheta = x1[15] # Update from solver result

        # Extract command (collective thrust and roll/pitch/yaw)
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