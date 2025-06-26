from casadi import SX, vertcat, mtimes, transpose, sqrt
import numpy as np

def create_tracking_cost_function(model, q_c=10.0, q_l=5.0):
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


    cost_y_expr = vertcat(e_c, e_l)
    cost_y_expr_e = vertcat(e_c,e_l)


    return cost_y_expr, cost_y_expr_e

