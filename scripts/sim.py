"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING
import fire
import gymnasium
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
import numpy as np


from lsy_drone_racing.utils import  load_config, load_controller, draw_gates, draw_point, draw_obstacles, generate_parallel_lines,draw_line
from lsy_drone_racing.utils.visualizer import SimVisualizer

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore", message="Explicitly requested dtype float64.*")
import time

### ========== logger ==========================
from datetime import datetime
import csv       # falls du CSV statt NPY willst
### ============================================


def simulate(
    config: str = "level0.toml",
    controller: str | None = None,
    n_runs: int = 1,
    gui: bool | None = None,
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
        n_runs: The number of episode
        gui: Enable/disable the simulation GUI.

    Returns:
        A list of episode times.
    """
    # Load configuration and check if firmare should be used.
    config = load_config(Path(__file__).parents[1] / "config" / config)

    ### ========== logger ==========================
    sim_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = Path(__file__).parents[1] / "logs" / sim_start
    log_root.mkdir(parents=True, exist_ok=True)
    ### ============================================

    if gui is None:
        gui = config.sim.gui
    else:
        config.sim.gui = gui
    # Load the controller module
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)  # This returns a class, not an instance
    # Create the racing environment
    env: DroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        # seed=config.env.seed,
        seed=int(time.time()) % 100000,
    )
    env = JaxToNumpy(env)

    ep_times = []
    try:
        for _ in range(n_runs):  # Run n_runs episodes with the controller
            obs, info = env.reset()
            controller: Controller = controller_cls(obs, info, config)

    #===========================================================================================
            # --- Get the planned path from the controller ---
            all_trajectories = [controller.get_trajectory()]
            
            # --- Prepare storage for the actually flown path ---
            flown_positions: list[np.ndarray] = []
            
            # --- Zur Erkennung von Gate-Positionsänderungen ---
            last_gates_positions = {}  # Dictionary mit gate_id: position zur Speicherung der letzten bekannten Positionen
            gate_update_points = []  # Liste für die Positionen, an denen Gate-Positionsänderungen erkannt wurden
            
            # --- Zur Erkennung von Obstacle-Positionsänderungen ---
            last_obstacles_positions = {}  # Dictionary mit obstacle_idx: position zur Speicherung der letzten bekannten Positionen
            obstacle_update_points = []  # Liste für die Positionen, an denen Obstacle-Positionsänderungen erkannt wurden
    #==========================================================================================
            i = 0
            fps = 60

            env.unwrapped.sim.max_visual_geom = 5_000

            # =============== logger =========================
            prev_go_values = None                 # voriger Snapshot (Liste[float])
            go_file       = log_root / f"run_{len(ep_times):03d}_gates_obst.csv"
            go_header_written = False             # Flag, damit Kopfzeile nur 1× kommt
            # =================================================

            while True:

                
                curr_time = i / config.env.freq
                ### ============== logger =========================
                run_log = controller.get_xcurrent_log()

                if run_log:                                      # nur speichern, wenn Daten da sind
                    num_states = len(run_log[0][1])
                    log_file = log_root / f"run_{len(ep_times):03d}_xcurrent.csv"
                    with log_file.open("w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["t", *[f"x{i}" for i in range(num_states)]])
                        for t, x in run_log:
                            writer.writerow([t, *x])
                else:
                    logger.warning("Kein xcurrent‑Log für diesen Run gesammelt.")
                                # ========== Gate/Obstacle Snapshot ==========
                g_pos  = obs["gates_pos"].flatten()
                g_quat = obs["gates_quat"].flatten()
                o_pos  = obs["obstacles_pos"].flatten()

                # 1) runden (2 Dez.) und in Liste verwandeln
                curr_go_values = [round(v, 3) for v in
                                np.hstack((g_pos, g_quat, o_pos)).tolist()]

                # 2) Vergleich & evtl. Schreiben
                if prev_go_values is None or curr_go_values != prev_go_values:
                    if not go_header_written:
                        header = ["t"]
                        n_gates = len(obs["gates_pos"])
                        for g in range(n_gates):
                            header += [f"g{g}_{k}" for k in ("x","y","z","qx","qy","qz","qw")]
                        n_obst = len(obs["obstacles_pos"])
                        for o in range(n_obst):
                            header += [f"o{o}_{k}" for k in ("x","y","z")]

                        with go_file.open("w", newline="") as f:
                            csv.writer(f).writerow(header)
                        go_header_written = True

                    with go_file.open("a", newline="") as f:
                        csv.writer(f).writerow([f"{curr_time:.3f}", *[f"{v:.3f}" for v in curr_go_values]])

                    prev_go_values = curr_go_values

                ### ===================================================



                action = controller.compute_control(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                # Update the controller internal state and models.
                controller_finished = controller.step_callback(action, obs, reward, terminated, truncated, info)
                # Update and visualize the simulation
                SimVisualizer.update_visualization(env, obs, controller, config, all_trajectories, flown_positions, last_gates_positions, gate_update_points, last_obstacles_positions, obstacle_update_points)
                if config.sim.gui:
                    env.render()
                if terminated or truncated or controller_finished:
                    break
                i += 1



            # Log the final positions of gates and obstacles
            controller.episode_callback(curr_time)  # Update the controller internal state and models.
            log_episode_stats(obs, info, config, curr_time)
            controller.episode_reset()
            ep_times.append(curr_time if obs["target_gate"] == -1 else None)


    finally:
        # Sicherstellen, dass die Umgebung immer ordnungsgemäß geschlossen wird
        env.close()
        print("Umgebung erfolgreich geschlossen.")
        
    return ep_times





def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:  # The drone has passed the final gate
        gates_passed = len(config.env.track.gates)
    finished = gates_passed == len(config.env.track.gates)
    logger.info(
        f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {gates_passed}\n"
    )




if __name__ == "__main__":


    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    ep_times = fire.Fire(simulate, serialize=lambda _: None)
