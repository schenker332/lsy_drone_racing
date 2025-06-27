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
        self.N = 50                    # Number of discretization steps
        self.T_HORIZON = 1.5           # Time horizon in seconds
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
        ts = np.linspace(0, 1, np.shape(waypoints)[0])
        self.cs_x = CubicSpline(ts, waypoints[:, 0])
        self.cs_y = CubicSpline(ts, waypoints[:, 1])
        self.cs_z = CubicSpline(ts, waypoints[:, 2])

        self.theta = 0
        self.last_v_theta = 1/ (9 * self.dt * self.freq) 
        self.last_v_theta_cmd = 1 / (9 * self.dt * self.freq)  

        # ========================= old trajectory ==================== ###



        # ### ==================== new trajectory ==================== ###


        # theta_values, x_vals, y_vals, z_vals, _, _, _,_,_ = arc_length_parametrization(waypoints, num_samples=1000)

        # self.cs_x = CubicSpline(theta_values, x_vals)
        # self.cs_y = CubicSpline(theta_values, y_vals)
        # self.cs_z = CubicSpline(theta_values, z_vals)

        # self.theta = 0.0        # Fortschrittsstartwert


        # self.time = 8
        # self.last_v_theta = 1 / (self.time * self.dt * self.freq)
        # self.last_v_theta_cmd = 0

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

        if self.theta >= 1.2:
            self.finished = True
            
        # print_output(tick=self._tick, obs=obs, freq=self.config.env.freq)

        rpy = R.from_quat(obs["quat"]).as_euler("xyz", degrees=False)

        # Construct the current state vector for the MPC solver
        xcurrent = np.concatenate((obs["pos"], obs["vel"], rpy, [self.last_f_collective, self.last_f_cmd], self.last_rpy_cmd, [self.last_v_theta, self.last_v_theta_cmd]) )
        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)



        for j in range(self.N):
            theta_j = min(self.theta + j * self.last_v_theta_cmd * self.dt, 1.0)
            theta_j_next = min(theta_j + 0.0001, 1.0)

            xj = self.cs_x(theta_j)
            yj = self.cs_y(theta_j)
            zj = self.cs_z(theta_j)

            xj_next = self.cs_x(theta_j_next)
            yj_next = self.cs_y(theta_j_next)
            zj_next = self.cs_z(theta_j_next)

            p_ref = np.array([xj, yj, zj, xj_next, yj_next, zj_next])
            self.acados_ocp_solver.set(j, "p", p_ref)


        theta_N = min(self.theta + self.N * self.last_v_theta_cmd * self.dt, 1.0)
        theta_N_plus = min(theta_N + 0.0001, 1.0)

        xN = self.cs_x(theta_N)
        yN = self.cs_y(theta_N)
        zN = self.cs_z(theta_N)

        xN_next = self.cs_x(theta_N_plus)
        yN_next = self.cs_y(theta_N_plus)
        zN_next = self.cs_z(theta_N_plus)

        p_ref_N = np.array([xN, yN, zN, xN_next, yN_next, zN_next])
        self.acados_ocp_solver.set(self.N, "p", p_ref_N)







        ## =================== new trajectory + new cost function ==================== ###
        # for j in range(self.N):
        #     theta_j = min(self.theta + j * self.last_v_theta_cmd * self.dt, 1.0)
        #     theta_j_next = min(self.theta + (j + 1) * self.last_v_theta_cmd * self.dt, 1.0)

        #     xj = self.cs_x(theta_j)
        #     yj = self.cs_y(theta_j)
        #     zj = self.cs_z(theta_j)

        #     xj_next = self.cs_x(theta_j_next)
        #     yj_next = self.cs_y(theta_j_next)
        #     zj_next = self.cs_z(theta_j_next)

        #     p_ref = np.array([xj, yj, zj, xj_next, yj_next, zj_next])
        #     self.acados_ocp_solver.set(j, "p", p_ref)


        # theta_N = min(self.theta + self.N * self.last_v_theta * self.dt, 1.0)
        # theta_N_plus = min(self.theta + (self.N + 1) * self.last_v_theta * self.dt, 1.0)

        # xN = self.cs_x(theta_N)
        # yN = self.cs_y(theta_N)
        # zN = self.cs_z(theta_N)

        # xN_next = self.cs_x(theta_N_plus)
        # yN_next = self.cs_y(theta_N_plus)
        # zN_next = self.cs_z(theta_N_plus)

        # p_ref_N = np.array([xN, yN, zN, xN_next, yN_next, zN_next])
        # self.acados_ocp_solver.set(self.N, "p", p_ref_N)
        # ### ================================================================ ###




        # Solve the MPC problem
        self.acados_ocp_solver.solve()

        x1 = self.acados_ocp_solver.get(1, "x")

        # print x with their names
        state_names = ["px", "py", "pz", "vx", "vy", "vz", "roll", "pitch", "yaw",
                       "f_collective", "f_collective_cmd", "r_cmd", "p_cmd", "y_cmd",
                       "v_theta", "v_theta_cmd"]
        for name, value in zip(state_names, x1):
            print(f"{name}: {value}")
        #print =====
        print("=" * 20)


        # total_cost = self.acados_ocp_solver.get_cost()
        # print(f"Solver Kosten: {total_cost:.4f}")
    
        # Apply low-pass filter to thrust for smoother control
        w = 1 / self.config.env.freq / self.dt
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]  

        # self.last_v_theta = 
        self.last_v_theta_cmd = x1[15]
        self.theta += self.last_v_theta_cmd * self.dt  # Update progress


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