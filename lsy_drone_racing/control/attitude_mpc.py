"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints
from typing import TYPE_CHECKING
import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, sin, vertcat
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control import Controller
if TYPE_CHECKING:
    from numpy.typing import NDArray






def export_quadrotor_ode_model() -> AcadosModel:
    """Symbolic Quadrotor Model."""
    # Define name of solver to be used in script
    model_name = "lsy_example_mpc"

    # Define Gravitational Acceleration
    GRAVITY = 9.806

    # Sys ID Params
    params_pitch_rate = [-6.003842038081178, 6.213752925707588]
    params_roll_rate = [-3.960889336015948, 4.078293254657104]
    params_yaw_rate = [-0.005347588299390372, 0.0]
    params_acc = [20.907574256269616, 3.653687545690674]

    """Model setting"""
    # define basic variables in state and input vector
    px = MX.sym("px")  # 0
    py = MX.sym("py")  # 1
    pz = MX.sym("pz")  # 2
    vx = MX.sym("vx")  # 3
    vy = MX.sym("vy")  # 4
    vz = MX.sym("vz")  # 5
    roll = MX.sym("r")  # 6
    pitch = MX.sym("p")  # 7
    yaw = MX.sym("y")  # 8
    f_collective = MX.sym("f_collective")

    f_collective_cmd = MX.sym("f_collective_cmd")
    r_cmd = MX.sym("r_cmd")
    p_cmd = MX.sym("p_cmd")
    y_cmd = MX.sym("y_cmd")

    df_cmd = MX.sym("df_cmd")
    dr_cmd = MX.sym("dr_cmd")
    dp_cmd = MX.sym("dp_cmd")
    dy_cmd = MX.sym("dy_cmd")

    # define state and input vector
    states = vertcat(
        px,
        py,
        pz,
        vx,
        vy,
        vz,
        roll,
        pitch,
        yaw,
        f_collective,
        f_collective_cmd,
        r_cmd,
        p_cmd,
        y_cmd,
    )
    inputs = vertcat(df_cmd, dr_cmd, dp_cmd, dy_cmd)

    # Define nonlinear system dynamics
    f = vertcat(
        vx,
        vy,
        vz,
        (params_acc[0] * f_collective + params_acc[1])
        * (cos(roll) * sin(pitch) * cos(yaw) + sin(roll) * sin(yaw)),
        (params_acc[0] * f_collective + params_acc[1])
        * (cos(roll) * sin(pitch) * sin(yaw) - sin(roll) * cos(yaw)),
        (params_acc[0] * f_collective + params_acc[1]) * cos(roll) * cos(pitch) - GRAVITY,
        params_roll_rate[0] * roll + params_roll_rate[1] * r_cmd,
        params_pitch_rate[0] * pitch + params_pitch_rate[1] * p_cmd,
        params_yaw_rate[0] * yaw + params_yaw_rate[1] * y_cmd,
        10.0 * (f_collective_cmd - f_collective),
        df_cmd,
        dr_cmd,
        dp_cmd,
        dy_cmd,
    )

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = model_name
    model.f_expl_expr = f
    model.f_impl_expr = None
    model.x = states
    model.u = inputs

    return model




def create_ocp_solver(
    Tf: float, N: int, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # set model
    model = export_quadrotor_ode_model()
    ocp.model = model

    # Get Dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados: https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    Q = np.diag(
        [
            10.0,
            10.0,
            15.0,  # Position
            0.01,
            0.01,
            0.01,  # Velocity
            0.1,
            0.1,
            0.1,  # rpy
            0.01,
            0.01,  # f_collective, f_collective_cmd
            0.01,
            0.01,
            0.01,
        ]
    )  # rpy_cmd

    R = np.diag([0.01, 0.01, 0.01, 0.01])

    Q_e = Q.copy()

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[:nx, :] = np.eye(nx)  # Only select position states   18x14  I= 14x14 (oben)
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions   18x4  I= 4x4 (unten)
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:nx, :nx] = np.eye(nx)  # Only select position states  14x14  I= 14x14 
    ocp.cost.Vx_e = Vx_e

    # Set initial references (we will overwrite these later on to make the controller track the traj.)
    # ocp.cost.yref = np.zeros((ny, ))
    # ocp.cost.yref_e = np.zeros((ny_e, ))
    ocp.cost.yref = np.array(
        [1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    ocp.cost.yref_e = np.array(
        [1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0]
    )

    # Set State Constraints
    ocp.constraints.lbx = np.array([0.1, 0.1, -1.57, -1.57, -1.57])
    ocp.constraints.ubx = np.array([0.55, 0.55, 1.57, 1.57, 1.57])
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

    # Set Input Constraints
    # ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0. -10.0])
    # ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0])
    # ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI
    ocp.solver_options.tol = 1e-5

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="lsy_example_mpc.json", verbose=verbose)

    return acados_ocp_solver, ocp


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

        # Same waypoints as in the trajectory controller. Determined by trial and error.
        # waypoints = np.array(
        #     [
        #         [1.0, 1.5, 0.05],
        #         [0.8, 1.0, 0.2],
        #         [0.55, -0.3, 0.5], #1
        #         [0.2, -1.3, 0.65],
        #         [1.1, -0.85, 1.1], #2
        #         [0.2, 0.5, 0.65],
        #         [0.0, 1.2, 0.525],  #3
        #         [0.0, 1.2, 1.1],
        #         [-0.5, -0.1, 1.1], #4
        #         [-0.5, -0.5, 1.1],
        #     ]
        # )



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
            [-0.5,   0,  1.11],#gate4
            [-0.5, -0.2,1.11 ]



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

        # self._gate_to_wp_index = {
        #     0: 2,
        #     1: 4,
        #     2: 6,
        #     3: 8
        # }   


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
            yref = np.array(
                [
                    self.x_des[i + j],
                    self.y_des[i + j],
                    self.z_des[i + j],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.35,
                    0.35,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
            self.acados_ocp_solver.set(j, "yref", yref)

        yref_N = np.array(
            [
                self.x_des[i + self.N],
                self.y_des[i + self.N],
                self.z_des[i + self.N],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.35,
                0.35,
                0.0,
                0.0,
                0.0,
            ]
        )

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


    def episode_callback(self):

        t = np.linspace(0, self._des_completion_time, len(self.x_des))
        trajectory = CubicSpline(t, np.stack([self.x_des, self.y_des, self.z_des], axis=1))

        self._saved_trajectory.append({
            "flown_path": np.array(self._path_log),
            "trajectory": trajectory,
            "gates": self._info.get("gates_pos", []).copy(),
            "gates_quat": self._info.get("gates_quat", []).copy(),
            "obstacles": self._info.get("obstacles_pos", []).copy(),
            "time": self._tick / self.freq,
            "t_total": self._des_completion_time,
            "waypoints": self._waypoints.copy(),
            "gate_log": self._gate_log,
            "obstacle_log": self._obstacle_log
        })






    def episode_reset(self):
        self._plotted_once = False
        self._path_log = []
        self._tick = 0  # Wichtig für Zeitmessung und nächste Episode

