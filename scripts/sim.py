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



from lsy_drone_racing.utils import  load_config, load_controller, draw_gates, draw_point, draw_obstacles, generate_parallel_lines,draw_line
from lsy_drone_racing.utils.visualizer import SimVisualizer

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.rest.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore", message="Explicitly requested dtype float64.*")






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
        seed=config.env.seed, 
    )
    env = JaxToNumpy(env)
 #use this if everything shall be more randomized int(time.time()) % 100000

    ep_times = []
    try:
        for _ in range(n_runs):  # Run n_runs episodes with the controller
            obs, info = env.reset()
            default_mass = env.unwrapped.drone_mass  # Standard-Masse
            current_mass = env.unwrapped.sim.data.params.mass  # Randomisierte Masse
            mass_deviation = current_mass - default_mass  # Abweichung
            print(f"Drone mass - Default: {default_mass}, Deviation: {mass_deviation}, Total: {current_mass}")
            controller: Controller = controller_cls(obs, info, config)
            visualizer = SimVisualizer()
            i = 0

            env.unwrapped.sim.max_visual_geom = 5_000

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
            print(f"Drone mass - Default: {default_mass}, Deviation: {mass_deviation}, Total: {current_mass}")
            controller.episode_callback(curr_time)  # Update the controller internal state and models.
            log_episode_stats(obs, info, config, curr_time)
            controller.episode_reset()
            visualizer.reset_episode()  # Reset visualizer for next episode
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
