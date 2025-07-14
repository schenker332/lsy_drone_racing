from casadi import SX, vertcat, mtimes, sqrt, exp, sum1, reshape

def contour_and_lag_error(model):
    """
    Compute the contour (orthogonal) and lag (tangential) errors between the drone's position and the reference trajectory.

    Args:
        model: The acados model containing state (x) and parameter (p) vectors.

    Returns:
        e_c: Contour error (distance orthogonal to the trajectory)
        e_l: Lag error (distance along the trajectory tangent)
    """
    x = model.x
    p = model.p

    # 1. Extract drone position from state vector
    px, py, pz = x[0], x[1], x[2]
    pos = vertcat(px, py, pz)

    # 2. Extract reference points from parameter vector
    #    ref: current reference point on the trajectory
    #    ref_next: next reference point (for tangent calculation)
    x_ref, y_ref, z_ref = p[0], p[1], p[2]
    x_next, y_next, z_next = p[3], p[4], p[5]
    ref = vertcat(x_ref, y_ref, z_ref)
    ref_next = vertcat(x_next, y_next, z_next)

    # 3. Calculate the tangent vector of the trajectory at the reference point
    tangent = ref_next - ref
    eps = 1e-10  # Small value to avoid division by zero
    t_norm = sqrt(mtimes(tangent.T, tangent) + eps)  # Norm of tangent
    t_hat = tangent / t_norm  # Unit tangent vector

    # 4. Compute the error vector from reference to drone position
    e = pos - ref

    # 5. Contour error: component of e orthogonal to the trajectory
    I = SX.eye(3)  # 3x3 identity matrix
    P = I - mtimes(t_hat, t_hat.T)  # Projection matrix for orthogonal component
    e_c_vec = mtimes(P, e)  # Orthogonal error vector
    e_c = sqrt(mtimes(e_c_vec.T, e_c_vec) + eps)  # Norm of orthogonal error

    # 6. Lag error: component of e along the trajectory tangent
    e_l = mtimes(t_hat.T, e)  # Scalar projection onto tangent

    return e_c, e_l


def get_min_distance_to_trajectory(model):
    """
    Compute the minimum Euclidean distance from the drone's current position to a reference point on the trajectory.

    Args:
        model: The acados model containing state (x) and parameter (p) vectors.
            - p[7:10]: Reference point on the trajectory (x_ref_min, y_ref_min, z_ref_min)
            - x[0:3]: Current drone position (x_current, y_current, z_current)

    Returns:
        min_distance: The minimum distance from the drone to the reference trajectory point.
    """
    p = model.p
    x = model.x

    # 1. Extract the reference point on the trajectory from parameters
    x_ref_min, y_ref_min, z_ref_min = p[7], p[8], p[9]
    # 2. Extract the drone's current position from the state vector
    x_current, y_current, z_current = x[0], x[1], x[2]

    # 3. Compute the Euclidean distance between the drone and the reference point
    min_distance = sqrt((x_current - x_ref_min)**2 + (y_current - y_ref_min)**2 + (z_current - z_ref_min)**2)

    return min_distance


### not in use at the moment
def get_exponential_obstacle_cost(model,
                                  num_obstacles: int = 4,
                                  weight: float = 1.0,
                                  k: float = 3.0) -> SX:
    """
    Exponential penalty for proximity to all obstacles:
      cost = sum_i weight * exp(-k * d_i)
    with d_i = horizontal distance to obstacle i.
    """
    x = model.x
    p = model.p

    # 1) Hole Hindernis-Vektor aus p
    obs_vec = p[10:]                           # flach, Länge 3*N
    obs_mat = reshape(obs_vec, 3, num_obstacles)

    # 2) Drohnen-Position in xy
    drone_xy = vertcat(x[0], x[1])

    # 3) Für jedes Hindernis d_i und Kosten_i berechnen
    cost_list = []
    for i in range(num_obstacles):
        obs_xy = obs_mat[0:2, i]               # [x_i; y_i]
        d2     = sum1((drone_xy - obs_xy)**2)  # Abstand²
        d      = sqrt(d2 + 1e-9)               # Abstand
        cost_i = weight * exp(-k * d)          # exp-Bestrafung
        cost_list.append(cost_i)

    # 4) Summe aller Kosten
    cost_vec = vertcat(*cost_list)
    total_cost = sum1(cost_vec)
    return total_cost
