"""Launch script for the real race.

Usage:

python deploy.py <path/to/controller.py> <path/to/config.toml>
python deploy.py  --controller attitude_mpc.py --config level2.toml



"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import rclpy

from lsy_drone_racing.utils import load_config, load_controller

## ========== logger ==========================
from datetime import datetime
import csv  
import numpy as np
### ============================================

if TYPE_CHECKING:
    from lsy_drone_racing.envs.real_race_env import RealDroneRaceEnv

logger = logging.getLogger(__name__)


def main(config: str = "level2.toml", controller: str | None = None):
    """Deployment script to run the controller on the real drone.

    Args:
        config: Path to the competition configuration. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
    """
    rclpy.init()
    config = load_config(Path(__file__).parents[1] / "config" / config)

    ### ========== logger ==========================
    sim_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = Path(__file__).parents[1] / "logs" / sim_start
    log_root.mkdir(parents=True, exist_ok=True)
    ### ============================================


    env: RealDroneRaceEnv = gymnasium.make(
        "RealDroneRacing-v0",
        drones=config.deploy.drones,
        freq=config.env.freq,
        track=config.env.track,
        randomizations=config.env.randomizations,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
    )
    try:
        options = {
            "check_drone_start_pos": config.deploy.check_drone_start_pos,
            "check_race_track": config.deploy.check_race_track,
            "real_track_objects": config.deploy.real_track_objects,
        }
        obs, info = env.reset(options=options)

        next_obs = obs  # Set next_obs to avoid errors when the loop never enters
        control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
        controller_path = control_path / (controller or config.controller.file)
        controller_cls = load_controller(controller_path)
        controller = controller_cls(obs, info, config)
        start_time = time.perf_counter()

        # =============== logger =========================
        prev_go_values = None                 # voriger Snapshot (Liste[float])
        go_file       = log_root / f"run_gates_obst.csv"
        go_header_written = False             # Flag, damit Kopfzeile nur 1× kommt
        # =================================================
        while rclpy.ok():
            t_loop = time.perf_counter()
            obs, info = env.unwrapped.obs(), env.unwrapped.info()
            obs = {k: v[0] for k, v in obs.items()}

            ### ============== logger ========================================================================
            run_log = controller.get_xcurrent_log()
            if run_log:                                      # nur speichern, wenn Daten da sind
                num_states = len(run_log[0][1])
                log_file = log_root / f"run_xcurrent.csv"
                with log_file.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["t", *[f"x{i}" for i in range(num_states)]])
                    for t, x in run_log:
                        writer.writerow([t, *x])
            else:
                pass

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
                    csv.writer(f).writerow([f"{t_loop:.3f}", *[f"{v:.3f}" for v in curr_go_values]])

                prev_go_values = curr_go_values
            ### ==================================================================================================



            action = controller.compute_control(obs, info)
            next_obs, reward, terminated, truncated, info = env.step(action)
            controller_finished = controller.step_callback(
                action, next_obs, reward, terminated, truncated, info
            )
            if terminated or truncated or controller_finished:
                break
            if (dt := (time.perf_counter() - t_loop)) < (1 / config.env.freq):
                time.sleep(1 / config.env.freq - dt)
            else:
                exc = dt - 1 / config.env.freq
                logger.warning(f"Controller execution time exceeded loop frequency by {exc:.3f}s.")
        ep_time = time.perf_counter() - start_time
        finished_track = next_obs["target_gate"] == -1
        logger.info(f"Track time: {ep_time:.3f}s" if finished_track else "Task not completed")
    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logger.setLevel(logging.INFO)
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    fire.Fire(main)
