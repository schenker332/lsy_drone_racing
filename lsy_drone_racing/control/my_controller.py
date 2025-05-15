from __future__ import annotations 
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import CubicSpline
from lsy_drone_racing.control import Controller
if TYPE_CHECKING:
    from numpy.typing import NDArray

from scripts.plotting import plot_3d
from lsy_drone_racing.control.utils_debug import gate_intersection
# ganz oben im File ergänzen
import json
from pathlib import Path


class TrajectoryController(Controller):
    
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict) -> None:

        super().__init__(obs, info, config)
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

        self.t_total = 11
        t = np.linspace(0, self.t_total, len(waypoints))
        self.trajectory = CubicSpline(t, waypoints)
        self._tick = 0
        self._freq = config.env.freq
        self._finished = False


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



    

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
    
        # print_output(obs, self._tick, self._freq)  # Ausgabe der Sensoren
        # == update of the gates ==
        for gate_id, seen in enumerate(obs["gates_visited"]):
            if seen and gate_id not in self._added_gates:
                idx = self._gate_to_wp_index.get(gate_id)
                self._waypoints[idx] = obs["gates_pos"][gate_id].copy()
                self._added_gates.add(gate_id)
                t = np.linspace(0, self.t_total, len(self._waypoints))
                self.trajectory = CubicSpline(t, self._waypoints)



        tau = min(self._tick / self._freq, self.t_total)
        target_pos = self.trajectory(tau)
        if tau == self.t_total:  # Maximum duration reached
            self._finished = True
        return np.concatenate((target_pos, np.zeros(10)), dtype=np.float32)
    
     

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
        return self._finished



    def episode_callback(self):

        self._saved_trajectory.append({
            "flown_path": np.array(self._path_log),
            "trajectory": self.trajectory,
            "gates": self._info.get("gates_pos", []).copy(),
            "gates_quat": self._info.get("gates_quat", []).copy(),
            "obstacles": self._info.get("obstacles_pos", []).copy(),
            "time": self._tick / self._freq,
            "t_total": self.t_total,
            "waypoints": self._waypoints.copy(),
            "gate_log": self._gate_log,
            "obstacle_log": self._obstacle_log
        })

        plot_3d(self._saved_trajectory[-1])



    def episode_reset(self):
        self._plotted_once = False
        self._path_log = []
        self._tick = 0  # Wichtig für Zeitmessung und nächste Episode


