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



