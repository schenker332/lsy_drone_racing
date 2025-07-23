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
import csv
import tomli
from lsy_drone_racing.utils import  load_config, load_controller
from lsy_drone_racing.utils.visualizer import SimVisualizer

if TYPE_CHECKING:
    

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore", message="Explicitly requested dtype float64.*")

from ml_collections import ConfigDict

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
    config = load_config(config_dir / config)
    config.mpc = load_mpc_config(config_dir)

    if gui is not None:
        config.sim.gui = gui

    control_path   = project_root / "lsy_drone_racing" / "control"
    ctrl_filename  = controller or config.controller.file
    controller_cls = load_controller(control_path / ctrl_filename)

    
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

    try:
        for _ in range(n_runs):

            obs, info = env.reset()


            controller: Controller = controller_cls(obs, info, config)
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
    finally:
        env.close()

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
