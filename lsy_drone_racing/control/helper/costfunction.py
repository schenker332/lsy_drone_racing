from casadi import SX, vertcat, mtimes, sqrt, exp, sum1, reshape

def contour_and_lag_error(model):
    x = model.x
    p = model.p

    # Position der Drohne
    px, py, pz = x[0], x[1], x[2]
    pos = vertcat(px, py, pz)

    # Referenzpunkte
    x_ref, y_ref, z_ref = p[0], p[1], p[2]
    x_next, y_next, z_next = p[3], p[4], p[5]
    ref = vertcat(x_ref, y_ref, z_ref)
    ref_next = vertcat(x_next, y_next, z_next)

    # Tangentenvektor
    tangent = ref_next - ref
    eps = 1e-10
    t_norm = sqrt(mtimes(tangent.T, tangent) + eps)
    t_hat = tangent / t_norm

    # Fehlervektor
    e = pos - ref

    # Contour Error (orthogonal zur Trajektorie)
    I = SX.eye(3)
    P = I - mtimes(t_hat, t_hat.T)
    e_c_vec = mtimes(P, e)
    e_c = sqrt(mtimes(e_c_vec.T, e_c_vec) + eps)

    # Lag Error (entlang der Trajektorie)
    e_l = mtimes(t_hat.T, e)


    return e_c, e_l


def get_min_distance_to_trajectory(model):
    p = model.p
    x = model.x
    x_ref_min, y_ref_min, z_ref_min = p[7], p[8], p[9]

    # Minimum distance to trajectory
    min_distance = sqrt((x[0] - x_ref_min)**2 + (x[1] - y_ref_min)**2 + (x[2] - z_ref_min)**2)

    return min_distance



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
