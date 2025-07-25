"""Utility module."""
from __future__ import annotations
import importlib.util
import inspect
import logging
import sys
from typing import TYPE_CHECKING, Type
import mujoco
import numpy as np
import toml
from ml_collections import ConfigDict
from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control.controller import Controller
from typing import List 

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from numpy.typing import NDArray

    from lsy_drone_racing.envs.race_core import RaceCoreEnv


logger = logging.getLogger(__name__)


def load_controller(path: Path) -> Type[Controller]:
    """Load the controller module from the given path and return the Controller class.

    Args:
        path: Path to the controller module.
    """
    assert path.exists(), f"Controller file not found: {path}"
    assert path.is_file(), f"Controller path is not a file: {path}"
    spec = importlib.util.spec_from_file_location("controller", path)
    controller_module = importlib.util.module_from_spec(spec)
    sys.modules["controller"] = controller_module
    spec.loader.exec_module(controller_module)

    def filter(mod: Any) -> bool:
        """Filter function to identify valid controller classes.

        Args:
            mod: Any attribute of the controller module to be checked.
        """
        subcls = inspect.isclass(mod) and issubclass(mod, Controller)
        return subcls and mod.__module__ == controller_module.__name__

    controllers = inspect.getmembers(controller_module, filter)
    controllers = [c for _, c in controllers if issubclass(c, Controller)]
    assert len(controllers) > 0, f"No controller found in {path}. Have you subclassed Controller?"
    assert len(controllers) == 1, f"Multiple controllers found in {path}. Only one is allowed."
    controller_module.Controller = controllers[0]
    assert issubclass(controller_module.Controller, Controller)

    try:
        return controller_module.Controller
    except ImportError as e:
        raise e


def load_config(path: Path) -> ConfigDict:
    """Load the race config file.

    Args:
        path: Path to the config file.

    Returns:
        The configuration.
    """
    assert path.exists(), f"Configuration file not found: {path}"
    assert path.suffix == ".toml", f"Configuration file has to be a TOML file: {path}"

    with open(path, "r") as f:
        return ConfigDict(toml.load(f))
    



def draw_point(
    env: RaceCoreEnv,
    point: NDArray,  # shape (3,)
    size: float = 0.05,
    rgba: NDArray | None = None,
):
    """Draw a spherical marker at a given 3D point.

    Args:
        env: The drone racing environment.
        point: np.array([x, y, z]) position of the point.
        size: Radius of the sphere marker (in meters).
        rgba: Optional RGBA color (default: opaque green).
    """
    assert point.shape == (3,), "Point must be a 3D coordinate"

    sim = env.unwrapped.sim
    if sim.viewer is None:
        return
    viewer = sim.viewer.viewer

    if rgba is None:
        rgba = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)  # opaque green

    size_vec = np.array([size, size, size], dtype=np.float32)
    # identity rotation for a sphere
    mat = np.eye(3, dtype=np.float32).reshape(-1)
    viewer.add_marker(
        type=mujoco.mjtGeom.mjGEOM_SPHERE, size=size_vec, pos=point, mat=mat, rgba=rgba
    )


def draw_line(
    env: RaceCoreEnv,
    points: NDArray,
    rgba: NDArray | None = None,
    min_size: float = 3.0,
    max_size: float = 3.0,
):
    """Draw a line into the simulation.

    Args:
        env: The drone racing environment.
        points: An array of [N, 3] points that make up the line.
        rgba: The color of the line.
        min_size: The minimum line size. We linearly interpolate the size from min_size to max_size.
        max_size: The maximum line size.

    Note:
        This function has to be called every time before the env.render() step.
    """
    assert points.ndim == 2, f"Expected array of [N, 3] points, got Array of shape {points.shape}"
    assert points.shape[-1] == 3, f"Points must be 3D, are {points.shape[-1]}"
    sim = env.unwrapped.sim
    if sim.viewer is None:  # Do not attempt to add markers if viewer is still None
        return
    if sim.max_visual_geom < points.shape[0]:
        raise RuntimeError("Attempted to draw too many lines. Try to increase Sim.max_visual_geom")
    viewer = sim.viewer.viewer
    sizes = np.zeros_like(points)[:-1, :]
    sizes[:, 2] = np.linalg.norm(points[1:] - points[:-1], axis=-1)
    sizes[:, :2] = np.linspace(min_size, max_size, len(sizes))[..., None]
    if rgba is None:
        rgba = np.array([1.0, 0, 0, 1])
    mats = _rotation_matrix_from_points(points[:-1], points[1:]).as_matrix().reshape(-1, 9)
    for i in range(len(points) - 1):
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_LINE, size=sizes[i], pos=points[i], mat=mats[i], rgba=rgba
        )


def draw_box(
    env: RaceCoreEnv,
    corner_min: NDArray,  # shape (3,)
    corner_max: NDArray,  # shape (3,)
    rgba: NDArray | None = None,
):
    """
    Draw a filled transparent 3D box from two opposite corners.

    Args:
        env: The drone racing environment.
        corner_min: np.array([x_min, y_min, z_min])
        corner_max: np.array([x_max, y_max, z_max])
        rgba: Optional RGBA color (default: semi-transparent red).
    """
    assert corner_min.shape == (3,) and corner_max.shape == (3,), "Each corner must be shape (3,)"

    sim = env.unwrapped.sim
    if sim.viewer is None:
        return
    viewer = sim.viewer.viewer

    if rgba is None:
        rgba = np.array([1.0, 0.0, 0.0, 0.3])  # semi-transparent red

    # Compute center and half-size
    center = (corner_min + corner_max) / 2
    extents = (corner_max - corner_min) / 2  # half-size per axis

    viewer.add_marker(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=extents,
        pos=center,
        mat=np.eye(3).reshape(-1),  # identity rotation (AABB)
        rgba=rgba,
    )


def _rotation_matrix_from_points(p1: NDArray, p2: NDArray) -> R:
    """Generate rotation matrices that align their z-axis to p2-p1."""
    z_axis = (v := p2 - p1) / np.linalg.norm(v, axis=-1, keepdims=True)
    random_vector = np.random.rand(*z_axis.shape)
    x_axis = (v := np.cross(random_vector, z_axis)) / np.linalg.norm(v, axis=-1, keepdims=True)
    y_axis = np.cross(z_axis, x_axis)
    return R.from_matrix(np.stack((x_axis, y_axis, z_axis), axis=-1))


def _quat_to_mat(q: NDArray) -> NDArray:
    """
    Converts a quaternion (x, y, z, w) into a 3 × 3 rotation matrix.
    
    Args:
        q: Quaternion as np.ndarray with shape (4,) in format (x, y, z, w)
        
    Returns:
        3x3 rotation matrix as np.ndarray
    """
    x, y, z, w = q
    # normalize as a safety measure
    n = np.linalg.norm(q)
    if n == 0:
        return np.eye(3)
    x, y, z, w = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def draw_gates(
    env: "RaceCoreEnv",
    gates_pos: NDArray,  # (N,3)
    gates_quat: NDArray,  # (N,4)  (x,y,z,w)
    half_extents: NDArray | None = None,  # Opening half-axes  (x/2 , y/2 , z/2)
    frame_thickness: float = None,  # Frame thickness in meters
    rgba_opening: NDArray | None = None,  # Color of the opening
    rgba_frame: NDArray | None = None,  # Color of the frame
) -> None:
    """
    Draws gate openings **and** colored frame beams.
    """
    # ------------------- Defaults ------------------------------------------------
    if half_extents is None:
        half_extents = np.array([0.225, 0.05, 0.225], dtype=np.float32)  # 0.45×0.45 opening
    if rgba_opening is None:
        rgba_opening = np.array([0.0, 0.4, 1.0, 0.0], dtype=np.float32)  # semi-transparent
    if rgba_frame is None:
        rgba_frame = np.array([1.0, 0.0, 0.0, 0.9], dtype=np.float32)  # opaque red

    sim = env.unwrapped.sim
    if sim.viewer is None:  # Headless
        return
    viewer = sim.viewer.viewer

    # ------------- Geometry parameters ------------------------------------------
    w, d, h = half_extents * 2  # full opening width/depth/height
    t = frame_thickness  # beam thickness (full)
    d_half = half_extents[1]  # half depth Y

    # Half edge lengths of the four beams
    size_vert = np.array([t / 2, d_half, (h + 2 * t) / 2], dtype=np.float32)
    size_horiz = np.array([(w + 2 * t) / 2, d_half, t / 2], dtype=np.float32)

    # Local offsets of the beam centers
    offs_left = np.array([-(w / 2 + t / 2), 0.0, 0.0], dtype=np.float32)
    offs_right = -offs_left
    offs_bottom = np.array([0.0, 0.0, -(h / 2 + t / 2)], dtype=np.float32)
    offs_top = -offs_bottom
    offsets = [
        (offs_left, size_vert),
        (offs_right, size_vert),
        (offs_bottom, size_horiz),
        (offs_top, size_horiz),
    ]

    # ------------- Rendering-Loop ----------------------------------------------
    for pos, q in zip(gates_pos, gates_quat):
        R = _quat_to_mat(q)

        # 1) Opening
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half_extents,
            pos=pos,
            mat=R.reshape(-1),
            rgba=rgba_opening,
        )

        # 2) Four frame beams
        for off_local, size in offsets:
            off_world = R @ off_local
            viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=size,
                pos=pos + off_world,
                mat=R.reshape(-1),
                rgba=rgba_frame,
            )


def draw_obstacles(
    env: Any,  # RaceCoreEnv
    obstacles_pos: NDArray,
    width: float = 0.3,       # Width of the obstacle (x-axis)
    depth: float = 0.3,       # Depth of the obstacle (y-axis)
    height: float = 0.3,      # Height of the obstacle (z-axis)
    position_top: bool = True, # If True: Position is top center; otherwise center
    rgba: NDArray | None = None,
):
    """Draw obstacles as boxes at their positions.

    Args:
        env: The drone racing environment.
        obstacles_pos: Array of obstacle positions, shape (n_obstacles, 3).
        width: Width of the obstacle box in meters (x-axis).
        depth: Depth of the obstacle box in meters (y-axis).
        height: Height of the obstacle box in meters (z-axis).
        position_top: If True, the position is the top center of the box,
                     if False, the position is the center of the box.
        rgba: Optional RGBA color (default: semi-transparent red).
    """
    sim = env.unwrapped.sim
    if sim.viewer is None:
        return
    viewer = sim.viewer.viewer

    if rgba is None:
        rgba = np.array([1.0, 0.0, 0.0, 0.7], dtype=np.float32)  # semi-transparent red

    # Half size, since MuJoCo expects half-axes of the box
    half_width = width / 2.0
    half_depth = depth / 2.0
    half_height = height / 2.0
    
    # Create the size vectors for the box
    half_size = np.array([half_width, half_depth, half_height], dtype=np.float32)
    
    # Identity matrix for rotation (no rotation)
    mat = np.eye(3, dtype=np.float32).reshape(-1)

    for pos in obstacles_pos:
        # Determine the actual position of the box center
        draw_pos = pos.copy()
        
        if position_top:
            # If position_top=True: Move the center downward by half the height
            # This makes the provided position the top center
            draw_pos[2] -= half_height
        
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_BOX, 
            size=half_size,  # Half-axes of the box
            pos=draw_pos,    # Position adjusted for position_top
            mat=mat, 
            rgba=rgba
        )




def generate_parallel_lines(trajectory: np.ndarray, radius: float = 0.3, num_lines: int = 16) -> List[np.ndarray]:
    """
    Generate short parallel line segments arranged in a circle around the trajectory.
    Each segment is oriented along the trajectory and offset in a radial direction.
    This is used to visualize constraints or boundaries around a path.
    
    Args:
        trajectory: np.ndarray of shape (N, 3), the main trajectory points in 3D space.
        radius: Distance from center line to each line segment (in meters).
        num_lines: Number of parallel lines per trajectory point, evenly distributed in a circle.

    Returns:
        List of np.ndarray of shape (2, 3), representing start and end of each line segment.
        Each line is a small segment oriented along the trajectory direction.
    """
    if len(trajectory) < 2:
        return []

    directions = trajectory[1:] - trajectory[:-1]
    directions = np.vstack([directions, directions[-1]])
    lines = []

    for i, (point, direction) in enumerate(zip(trajectory, directions)):
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        direction = direction / norm
        if abs(direction[2]) < 0.99:
            ortho1 = np.cross(direction, [0, 0, 1])
        else:
            ortho1 = np.cross(direction, [0, 1, 0])
        ortho1 /= np.linalg.norm(ortho1)
        ortho2 = np.cross(direction, ortho1)

        for j in range(num_lines):
            angle = 2 * np.pi * j / num_lines
            offset = radius * (np.cos(angle) * ortho1 + np.sin(angle) * ortho2)
            p1 = point + offset
            p2 = point + offset + 0.1 * direction
            lines.append(np.stack([p1, p2]))

    return lines


def draw_tube_lines(env: Any, trajectory: np.ndarray, radius: float = 0.3, num_lines: int = 16):
    """
    Draws a set of parallel lines around a given trajectory segment to create a tube-like appearance.
    This can be used to visualize constraints or safe flight corridors around a trajectory.
    It additionally generates a sphere over the first segment to indicate the relative value of the maximal cost relative a cost of to 150.

    Args:
        env: Simulation environment with rendering capabilities.
        trajectory: The main trajectory (e.g., prediction horizon) as np.ndarray of shape (N, 3).
        radius: Radius of the virtual tube in meters.
        num_lines: Number of parallel lines to draw per point, evenly distributed around the circle.
            Higher values give a more continuous tube appearance but use more rendering resources.
    """
    lines = generate_parallel_lines(trajectory, radius=radius, num_lines=num_lines)
    for line in lines:
        draw_line(env, line, rgba=np.array([1.0, 1.0, 0.0, 0.4]), min_size=1.0, max_size=1.0)


def visualize_cost_weights(env, controller):
    """Visualize cost weights as orthogonal bars along the trajectory.
    
    This function draws bars perpendicular to the trajectory at the MPC horizon stages,
    with bar length proportional to the cost weight at that position. This shows
    the actual weights used by the MPC at each stage of the prediction horizon.
    
    Args:
        env: The simulation environment for drawing
        controller: The MPC controller containing the trajectory and weight functions
    """
    # --- Configuration ---
    gate_width = 0.4  # Reference width for normalization (gate opening is 0.4m)
    bar_thickness = 3.0  # Line thickness for the bars
    min_bar_length = 0.05  # Minimum bar length for visibility
    
    # --- Get MPC horizon data ---
    N = controller.N  # Number of stages in the horizon
    dt = controller.dt  # Time step
    current_theta = controller.theta
    v_theta = controller.v_theta
    
    # --- Collect weights from the actual MPC horizon ---
    horizon_thetas = []
    horizon_weights = []
    
    for j in range(N + 1):  # Same loop as in compute_control
        # Compute the theta value for the current stage (same as in MPC)
        theta_j = min(current_theta + j * v_theta * dt, 1.0)
        weight_j = controller.weight_for_theta(theta_j)
        
        horizon_thetas.append(theta_j)
        horizon_weights.append(weight_j)
    
    # Convert to numpy arrays for easier processing
    horizon_thetas = np.array(horizon_thetas)
    horizon_weights = np.array(horizon_weights)
    
    # --- Normalize weights ---
    max_weight = np.max(horizon_weights)
    min_weight = np.min(horizon_weights)
    
    # # Debug output
    # print(f"MPC Horizon Weights: min={min_weight:.3f}, max={max_weight:.3f}, current weight={horizon_weights[0]:.3f} stages={len(horizon_weights)}")

    # Avoid division by zero
    if max_weight == min_weight:
        max_weight = min_weight + 1
    
    # --- Draw bars at each MPC stage ---
    bars_drawn = 0
    for j, (theta, weight) in enumerate(zip(horizon_thetas, horizon_weights)):
        # Get reference point on trajectory
        ref_point = np.array([
            controller.cs_x(theta), 
            controller.cs_y(theta), 
            controller.cs_z(theta)
        ])
        
        # Calculate tangent vector
        theta_next = min(theta + 0.001, 1.0)
        next_point = np.array([
            controller.cs_x(theta_next),
            controller.cs_y(theta_next), 
            controller.cs_z(theta_next)
        ])
        
        tangent = next_point - ref_point
        tangent_norm = np.linalg.norm(tangent)
        
        # Skip if tangent is too small
        if tangent_norm < 1e-10:
            continue
            
        t_hat = tangent / tangent_norm
        
        # Find orthogonal vector (perpendicular to trajectory)
        if abs(t_hat[2]) < 0.9:
            ortho = np.cross(t_hat, np.array([0, 0, 1]))
        else:
            ortho = np.cross(t_hat, np.array([0, 1, 0]))
        
        ortho_norm = np.linalg.norm(ortho)
        if ortho_norm < 1e-10:
            continue
            
        ortho = ortho / ortho_norm
        
        # Normalize weight to bar length
        normalized_weight = (weight - min_weight) / (max_weight - min_weight)
        bar_length = max(min_bar_length, normalized_weight * gate_width)
        
        # Calculate bar endpoints (centered on trajectory)
        half_length = bar_length / 2
        p1 = ref_point - half_length * ortho
        p2 = ref_point + half_length * ortho
        
        # Skip if points are too close
        if np.linalg.norm(p2 - p1) < 1e-8:
            continue
        
        # Enhanced color mapping based on stage and weight
        if j == 0:  # Current stage - special color
            rgba = np.array([1.0, 1.0, 0.0, 0.9])  # Yellow for current position
        else:
            # Color mapping: red for high weights, blue for low weights
            color_intensity = normalized_weight
            rgba = np.array([
                color_intensity,      # Red component
                0.1,                  # Green component (low for contrast)
                1.0 - color_intensity, # Blue component (inverse)
                0.8                   # Alpha (transparency)
            ])
        
        # # Debug output for first few bars
        # if bars_drawn < 5:
        #     print(f"Stage {j}: theta={theta:.3f}, weight={weight:.1f}, "
        #           f"normalized={normalized_weight:.3f}, bar_length={bar_length:.3f}")
        
        # Draw the bar
        draw_line(env, np.vstack([p1, p2]), 
                 rgba=rgba, 
                 min_size=bar_thickness, 
                 max_size=bar_thickness)    


        
        bars_drawn += 1

