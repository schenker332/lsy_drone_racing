from __future__ import annotations 
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control import Controller
if TYPE_CHECKING:
    from numpy.typing import NDArray
from lsy_drone_racing.control.plotting import plot_3d
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

        waypoints = np.array(
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
            [-0.5,   -0.1,  1.11],#gate4
     


        ])
        # Scale trajectory between 0 and 1
        ts = np.linspace(0, 1, np.shape(waypoints)[0])
        cs_x = CubicSpline(ts, waypoints[:, 0])
        cs_y = CubicSpline(ts, waypoints[:, 1])
        cs_z = CubicSpline(ts, waypoints[:, 2])




        self._des_completion_time = 7
        ts = np.linspace(0, 1, int(self.freq * self._des_completion_time))
        self.x_des = cs_x(ts)
        self.y_des = cs_y(ts)
        self.z_des = cs_z(ts)


        self.N = 50
        self.T_HORIZON = 1.5
        self.dt = self.T_HORIZON / self.N
        self.x_des = np.concatenate((self.x_des, [self.x_des[-1]] * (2 * self.N + 1)))
        self.y_des = np.concatenate((self.y_des, [self.y_des[-1]] * (2 * self.N + 1)))
        self.z_des = np.concatenate((self.z_des, [self.z_des[-1]] * (2 * self.N + 1)))


        self.acados_ocp_solver, self.ocp = create_ocp_solver(self.T_HORIZON, self.N)

        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.config = config
        self.finished = False




        self._info = info
        self._waypoints = waypoints
        self._path_log = [] 

        self._last_gate = []
        self._gate_log = []
        self._last_obstacle = []
        self._obstacle_log = []


        self._gate_to_wp_index = {
            0: 3,
            1: 6,
            2: 9,
            3: 13
        }   

        self._added_gates = set()
        self._saved_trajectory = []




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


        # ====== update waypoints =====
        for gate_id, seen in enumerate(obs["gates_visited"]):
            if seen and gate_id not in self._added_gates:
                idx = self._gate_to_wp_index.get(gate_id)
                self._waypoints[idx] = obs["gates_pos"][gate_id].copy()
                self._added_gates.add(gate_id)

                ts = np.linspace(0, 1, np.shape(self._waypoints)[0])
                cs_x = CubicSpline(ts, self._waypoints[:, 0])
                cs_y = CubicSpline(ts, self._waypoints[:, 1])
                cs_z = CubicSpline(ts, self._waypoints[:, 2])

  
                ts = np.linspace(0, 1, int(self.freq * self._des_completion_time))

                self.x_des = cs_x(ts)
                self.y_des = cs_y(ts)
                self.z_des = cs_z(ts)

                self.x_des = np.concatenate((self.x_des, [self.x_des[-1]] * (2 * self.N + 1)))
                self.y_des = np.concatenate((self.y_des, [self.y_des[-1]] * (2 * self.N + 1)))
                self.z_des = np.concatenate((self.z_des, [self.z_des[-1]] * (2 * self.N + 1)))




        for j in range(self.N):
            yref = np.array([self.x_des[i + j], self.y_des[i + j], self.z_des[i + j], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0])
            self.acados_ocp_solver.set(j, "yref", yref)


        yref_N = np.array([self.x_des[i + self.N], self.y_des[i + self.N], self.z_des[i + self.N], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0,0.0])
        self.acados_ocp_solver.set(self.N, "yref", yref_N)



        self.acados_ocp_solver.solve()
        x1 = self.acados_ocp_solver.get(1, "x")
        w = 1 / self.config.env.freq / self.dt
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_f_cmd = x1[10]
        self.last_rpy_cmd = x1[11:14]

        cmd = x1[10:14]
    
        return cmd



    def get_obs(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None):

        # ===== updated gates =====
        if not self._last_gate:
            for i,gate in enumerate(obs["gates_pos"]):
                self._last_gate.append(gate.copy())  # list: [1.gate][2.gate][][]
                self._gate_log.append([])
                self._gate_log[i].append((self._tick, gate.copy(), obs["gates_quat"][i].copy()))   # list: [(T1)(T2)(T3)]  [][][]


        for i, gate in enumerate(obs["gates_pos"]):
            last_pos = self._last_gate[i]
            if not np.allclose(gate, last_pos, atol= 1e-3):
                self._last_gate[i] = gate.copy()
                self._gate_log[i].append((self._tick, gate.copy(),obs["gates_quat"][i].copy()))


        # ===== updated obstacles =====
        if not self._last_obstacle:
            for i, obs_pos in enumerate(obs["obstacles_pos"]):
                self._last_obstacle.append(obs_pos.copy())
                self._obstacle_log.append([])
                self._obstacle_log[i].append((self._tick, obs_pos.copy()))

        for i, obs_pos in enumerate(obs["obstacles_pos"]):
            last_pos = self._last_obstacle[i]
            if not np.allclose(obs_pos, last_pos, atol=1e-4):
                self._last_obstacle[i] = obs_pos.copy()
                self._obstacle_log[i].append((self._tick, obs_pos.copy()))


    def step_callback(self,action: NDArray[np.floating],obs: dict[str, NDArray[np.floating]],reward: float,terminated: bool,truncated: bool,info: dict,) -> bool:
        self._tick += 1
        self._info = obs
        self._path_log.append(obs["pos"].copy())  # Position speichern
        self.get_obs(obs, info)
        return self.finished


    def episode_callback(self, curr_time: float = None):

        t = np.linspace(0, self._des_completion_time, len(self.x_des))
        trajectory = CubicSpline(t, np.stack([self.x_des, self.y_des, self.z_des], axis=1))



        self._saved_trajectory.append({
            "flown_path": np.array(self._path_log),
            "trajectory": trajectory,
            "gates": self._info.get("gates_pos", []).copy(),
            "gates_quat": self._info.get("gates_quat", []).copy(),
            "obstacles": self._info.get("obstacles_pos", []).copy(),
            "time": curr_time,
            "t_total": self._des_completion_time,
            "waypoints": self._waypoints.copy(),
            "gate_log": self._gate_log,
            "obstacle_log": self._obstacle_log
        })


        plot_3d(self._saved_trajectory[-1])



    def episode_reset(self):
        self._plotted_once = False
        self._path_log = []
        self._tick = 0  # Wichtig für Zeitmessung und nächste Episode



