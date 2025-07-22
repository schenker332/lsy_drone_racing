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
import time
from plots.plot_speed import plot_speed


from lsy_drone_racing.utils import  load_config, load_controller, draw_gates, draw_point, draw_obstacles, generate_parallel_lines,draw_line
from lsy_drone_racing.utils.visualizer import SimVisualizer

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.rest.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore", message="Explicitly requested dtype float64.*")

from datetime import datetime
import os
import csv
from lsy_drone_racing.control.helper.datalogger import DataLogger
from ml_collections import ConfigDict             # <<--- hier richtig importieren
### ======================= Logger ======================== ###
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
batch_dir = Path("logs") / f"batch_{timestamp}"
batch_dir.mkdir(parents=True, exist_ok=True)    

run_stats = []          # hier sammeln wir später Zeit + Gates pro Run
successful_times = []   # für die Erfolgs‑Durchschnittszeit
### ======================================================== ###
import tomli
def load_mpc_config(config_dir: Path) -> ConfigDict:
    """
    Lädt 'config/mpc.toml' und gibt den Unter-Abschnitt [mpc] zurück.
    """
    toml_path = config_dir / "mpc.toml"
    if not toml_path.exists():
        raise FileNotFoundError(f"mpc.toml nicht gefunden unter {toml_path}")
    with open(toml_path, "rb") as f:
        raw = tomli.load(f)
    return ConfigDict(raw["mpc"])  # jetzt funktioniert ConfigDict



def simulate(
    config: str = "level0.toml",
    controller: str | None = None,
    n_runs: int = 1,
    gui: bool | None = None,
    param_overrides: dict[str, float] | None = None,
    return_stats: bool = False
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
    project_root = Path(__file__).parents[1]
    config_dir   = project_root / "config"

    # 1) Umgebungs‑Config einmalig laden
    config = load_config(config_dir / config)

    # 2) MPC‑Parameter hinzufügen
    config.mpc = load_mpc_config(config_dir)

    # 3) GUI‑Override (falls vom CLI übergeben)
    if gui is not None:
        config.sim.gui = gui

    # 4) Controller‑Klasse ermitteln
    control_path   = project_root / "lsy_drone_racing" / "control"
    ctrl_filename  = controller or config.controller.file
    controller_cls = load_controller(control_path / ctrl_filename)

    # 1.2) Parameter‑Overrides anwenden
    if param_overrides:
        for k, v in param_overrides.items():
            if hasattr(config.mpc, k):
                setattr(config.mpc, k, v)
            else:
                raise KeyError(f"Unknown MPC parameter: {k}")

    
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
        # seed=int(time.time()) % 100000,
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)


    ep_times    = []
    total_gates = len(config.env.track.gates)
    try:
        for run_idx in range(1, n_runs + 1):          # ← jetzt mit Index

            obs, info = env.reset()

            default_mass   = env.unwrapped.drone_mass.item()          
            current_mass   = env.unwrapped.sim.data.params.mass.item()  
            mass_deviation = current_mass - default_mass
            print(f"Drone mass – Default: {default_mass:.6f}, Deviation: {mass_deviation:.6f}, Total: {current_mass:.6f}")

            controller: Controller = controller_cls(obs, info, config)

            ### ======================== Logger ======================== ###
            run_dir = batch_dir / f"run_{run_idx}"
            run_dir.mkdir(exist_ok=True)
            if getattr(controller, "logger", None):
                logger = controller.logger
                logger.run_dir = str(run_dir)
                logger.state_log_file = run_dir / "state_log.csv"
                logger.control_log_file = run_dir / "control_log.csv"
                logger.gates_log_file = run_dir / "final_gates.csv"
                logger.obstacles_log_file = run_dir / "final_obstacles.csv"
                logger._prepare_state_log()
                logger._prepare_control_log()
            ### ======================================================== ###

            visualizer = SimVisualizer()
            i = 0


            while True:
                curr_time = i / config.env.freq

                action = controller.compute_control(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                # Update the controller internal state and models.
                controller_finished = controller.step_callback(action, obs, reward, terminated, truncated, info)
                # Update and visualize the simulation
                visualizer.update_visualization(env, obs, controller)

                if config.sim.gui:
                    env.render()

                if terminated or truncated or controller_finished:
                    break
                i += 1


            log_episode_stats(obs, info, config, curr_time)
            controller.episode_reset()
            visualizer.reset_episode()  # Reset visualizer for next episode

            ### ======================== Logger ======================== ###
            if controller.logger:
                controller.logger.log_final_positions(
                    gates_pos=controller._info.get("gates_pos"),
                    obstacles_pos=controller._info.get("obstacles_pos")
                )
                controller.logger.close()

            # Wenn Episode fertig:
            gates_passed = obs["target_gate"]
            if gates_passed == -1:
                gates_passed = len(config.env.track.gates)

            run_stats.append({"run": run_idx,
                            "time": curr_time,
                            "gates_passed": gates_passed,
                            "mass": current_mass})

            if gates_passed == len(config.env.track.gates):
                successful_times.append(curr_time)



            ep_times.append(curr_time if gates_passed == total_gates else None)
            plot_speed(run_dir)
            ### ======================================================== ###


        ### ======================== Logger ======================== ###



        summary_file = batch_dir / "summary.csv"
        with open(summary_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run", "time_sec", "gates_passed", "mass"])

            for row in run_stats:
                writer.writerow([row["run"], f"{row['time']:.3f}", row["gates_passed"], f"{row['mass']:.3f}"])

            # Aggregierte Kennzahlen
            successful_runs = len(successful_times)
            avg_time = (sum(successful_times) / successful_runs) if successful_runs else None

            writer.writerow([])  # Leerzeile als Trenner
            writer.writerow(["successful_runs", successful_runs])
            writer.writerow(["avg_time_successful", f"{avg_time:.3f}" if avg_time else "n/a"])
        ### ======================================================== ###


    finally:
        # Sicherstellen, dass die Umgebung immer ordnungsgemäß geschlossen wird
        env.close()
        print("Umgebung erfolgreich geschlossen.")

        
    if return_stats:
        return run_stats
    else:
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
