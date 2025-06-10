from __future__ import annotations 
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control import Controller
if TYPE_CHECKING:
    from numpy.typing import NDArray
from lsy_drone_racing.control.create_ocp_solver import create_ocp_solver
from lsy_drone_racing.control.print_output import print_output



class MPController(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self.freq = config.env.freq
        self._tick = 0

        self.waypoints = np.array(
        [
            [1.0, 1.5, 0.3],
            [0.8, 1.0, 0.2],
            [0.7, 0.1, 0.4],
            [ 0.45, -0.5,   0.56],#gate1
            [0.2, -0.7, 0.65],
            [0.5, -1.5 , 0.8 ],
            [ 1,  -1.05,  1.11],#gate2
            [1.15,-0.75,1],
            [0.5, 0, 0.8],
            [0,   1,   0.56],#gate3
            [-0.1,1.2, 0.56],
            [-0.3, 1.2, 1.1],
            [-0.2,0.4, 1.1 ],
            [-0.45, 0.1, 1.11],
            [-0.5,   0,  1.11],#gate4
            [-0.5, -0.2,1.11 ],
        ])


        

        self.N = 50
        self.T_HORIZON = 1.5    
        self.dt = self.T_HORIZON / self.N



        self.acados_ocp_solver, self.ocp = create_ocp_solver(self.T_HORIZON, self.N)

        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.config = config
        self.finished = False
        self._info = info
        self._path_log = [] 





    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The collective thrust and orientation [t_des, r_des, p_des, y_des] as a numpy array.
        """
        i = min(self._tick, len(self.x_des) - 1)
        if self._tick > i:
            self.finished = True


        print_output(obs, self._tick, self.freq)
        q = obs["quat"]
        r = R.from_quat(q)
        # Convert to Euler angles in XYZ order
        rpy = r.as_euler("xyz", degrees=False)  # Set degrees=False for radians



        xcurrent = np.concatenate(
            (
                obs["pos"],
                obs["vel"],
                rpy,
                np.array([self.last_f_collective, self.last_f_cmd]),
                self.last_rpy_cmd,
            )
        )

        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)







        for j in range(self.N):
            idx = min(i + j, len(self.x_des) - 1)
            yref = np.array([self.x_des[idx], self.y_des[idx], self.z_des[idx], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.acados_ocp_solver.set(j, "yref", yref)

            p_ref = np.array([self.x_des[idx], self.y_des[idx], self.z_des[idx]])
            self.acados_ocp_solver.set(j, "p", p_ref)

        idx_N = min(i + self.N, len(self.x_des) - 1)
        yref_N = np.array([self.x_des[idx_N], self.y_des[idx_N], self.z_des[idx_N], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0])
        self.acados_ocp_solver.set(self.N, "yref", yref_N)
        p_ref_N = np.array([self.x_des[idx_N], self.y_des[idx_N], self.z_des[idx_N]])
        self.acados_ocp_solver.set(self.N, "p", p_ref_N)



        self.acados_ocp_solver.solve()
        x1 = self.acados_ocp_solver.get(1, "x")
        w = 1 / self.config.env.freq / self.dt
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]

        cmd = x1[10:14]
    
        return cmd





    def step_callback(self,action: NDArray[np.floating],obs: dict[str, NDArray[np.floating]],reward: float,terminated: bool,truncated: bool,info: dict,) -> bool:
        self._tick += 1
        self._info = obs
        return self.finished




    def episode_callback(self, curr_time: float = None):
        """Update controller internal state at each simulation step.
        
        Args:
            curr_time: Current simulation time in seconds.
        """
        # You can use curr_time to update your controller state if needed
        pass

    def episode_reset(self):
        self._plotted_once = False
        self._path_log = []
        self._tick = 0  # Wichtig für Zeitmessung und nächste Episode



    def get_trajectory(self) -> NDArray[np.floating]:
        """Get the trajectory points."""
                # Scale trajectory between 0 and 1
        ts = np.linspace(0, 1, np.shape(self.waypoints)[0])
        cs_x = CubicSpline(ts, self.waypoints[:, 0])
        cs_y = CubicSpline(ts, self.waypoints[:, 1])
        cs_z = CubicSpline(ts, self.waypoints[:, 2])


        self._des_completion_time = 4
        ts = np.linspace(0, 1, int(self.freq * self._des_completion_time))
        self.x_des = cs_x(ts)
        self.y_des = cs_y(ts)
        self.z_des = cs_z(ts)

        vis_s = np.linspace(0.0, 1.0, 700)  # Bereich [0,1] für konsistente Visualisierung
        traj_points = np.column_stack((cs_x(vis_s), cs_y(vis_s), cs_z(vis_s)))


        return traj_points
        





    def get_prediction_horizon(self) -> NDArray[np.floating]:
        """Get the predicted position trajectory for the planning horizon.
        
        Returns:
            Array of shape (N, 3) containing the predicted x,y,z positions
            for the next N timesteps in the planning horizon.
        """
        # Sammle die vorhergesagten Zustände für alle Schritte im Horizont
        horizon_positions = []
        for i in range(self.N):
            state = self.acados_ocp_solver.get(i, "x")
            # Die ersten drei Elemente des Zustands sind die x,y,z-Positionen
            pos = state[:3]
            horizon_positions.append(pos)
        
        return np.array(horizon_positions)
