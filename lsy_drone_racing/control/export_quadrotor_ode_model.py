from __future__ import annotations
import types
from acados_template import AcadosModel
from casadi import MX, cos, sin, vertcat, Function, SX

def export_quadrotor_ode_model():
    """Erstellt ein AcadosModel für den Quadrotor mit Tube-Constraint.
    
    Returns:
        tuple: (AcadosModel, constraint) - Das fertig konfigurierte Modell und Constraint-Objekt
    """
    # Erstelle Namespaces für Modell und Constraint
    model = types.SimpleNamespace()
    constraint = types.SimpleNamespace()
    
    # Setze den Modellnamen - wichtig für die JSON-Datei
    model.name = "lsy_example_mpc"

    # Physikalische Parameter
    GRAVITY = 9.806
    params_pitch_rate = [-6.003842038081178, 6.213752925707588]
    params_roll_rate = [-3.960889336015948, 4.078293254657104]
    params_yaw_rate = [-0.005347588299390372, 0.0]
    params_acc = [20.907574256269616, 3.653687545690674]
    
    # Tube-Constraint-Radius
    constraint.tube_radius = 0.3
    
    # Zustände definieren
    px = SX.sym("px")
    py = SX.sym("py")
    pz = SX.sym("pz")
    vx = SX.sym("vx")
    vy = SX.sym("vy")
    vz = SX.sym("vz")
    roll = SX.sym("r")
    pitch = SX.sym("p")
    yaw = SX.sym("y")
    f_collective = SX.sym("f_collective")
    f_collective_cmd = SX.sym("f_collective_cmd")
    r_cmd = SX.sym("r_cmd")
    p_cmd = SX.sym("p_cmd")
    y_cmd = SX.sym("y_cmd")
    
    # Eingaben definieren
    df_cmd = SX.sym("df_cmd")
    dr_cmd = SX.sym("dr_cmd")
    dp_cmd = SX.sym("dp_cmd")
    dy_cmd = SX.sym("dy_cmd")

    # Parameter (für Tube-Constraint)
    x_ref = SX.sym("x_ref")
    y_ref = SX.sym("y_ref")
    z_ref = SX.sym("z_ref")
    x_ref_next = SX.sym("x_ref_next")
    y_ref_next = SX.sym("y_ref_next")
    z_ref_next = SX.sym("z_ref_next")
    

    
    # Zustands- und Eingabevektoren
    states = vertcat(
        px, py, pz, vx, vy, vz, roll, pitch, yaw,
        f_collective, f_collective_cmd, r_cmd, p_cmd, y_cmd
    )
    controls = vertcat(df_cmd, dr_cmd, dp_cmd, dy_cmd)
    parameters = vertcat(x_ref, y_ref, z_ref, x_ref_next, y_ref_next, z_ref_next                     )
    
    # Modell mit Zuständen, Eingaben und Parametern
    model.x = states
    model.u = controls
    model.p = parameters

    # Systemdynamik
    f_expl = vertcat(
        vx,
        vy,
        vz,
        (params_acc[0] * f_collective + params_acc[1]) * (cos(roll) * sin(pitch) * cos(yaw) + sin(roll) * sin(yaw)),
        (params_acc[0] * f_collective + params_acc[1]) * (cos(roll) * sin(pitch) * sin(yaw) - sin(roll) * cos(yaw)),
        (params_acc[0] * f_collective + params_acc[1]) * cos(roll) * cos(pitch) - GRAVITY,
        params_roll_rate[0] * roll + params_roll_rate[1] * r_cmd,
        params_pitch_rate[0] * pitch + params_pitch_rate[1] * p_cmd,
        params_yaw_rate[0] * yaw + params_yaw_rate[1] * y_cmd,
        10.0 * (f_collective_cmd - f_collective),
        df_cmd,
        dr_cmd,
        dp_cmd,
        dy_cmd,
    )
    model.f_expl_expr = f_expl
    
    # Echter Tube-Constraint mit orthogonaler Distanz zur Trajektorielinie
    # Vector vom aktuellen Punkt zum Referenzpunkt
    dx = px - x_ref
    dy = py - y_ref
    dz = pz - z_ref

    # Vektor der Trajektorie (Richtung)
    traj_dir_x = x_ref_next - x_ref
    traj_dir_y = y_ref_next - y_ref
    traj_dir_z = z_ref_next - z_ref

    # Normieren (mit kleiner Epsilon-Zahl zur Vermeidung von Division durch Null)
    eps = 1e-10
    traj_norm = SX.sqrt(traj_dir_x**2 + traj_dir_y**2 + traj_dir_z**2 + eps)
    traj_dir_x = traj_dir_x / traj_norm
    traj_dir_y = traj_dir_y / traj_norm
    traj_dir_z = traj_dir_z / traj_norm

    # Projektion des Abstandsvektors auf die Trajektorienrichtung
    proj = dx*traj_dir_x + dy*traj_dir_y + dz*traj_dir_z

    # Orthogonale Komponente (die tatsächliche Entfernung vom "Tube")
    orth_dist_sq = dx**2 + dy**2 + dz**2 - proj**2
    # constraint.expr = orth_dist_sq  # ≤ tube_radius_sq
    

    constraint.shape = 1  # Dimension des Constraints (skalar)
    
    # Erstelle das AcadosModel
    acados_model = AcadosModel()
    acados_model.f_expl_expr = model.f_expl_expr
    acados_model.x = model.x
    acados_model.u = model.u
    acados_model.p = model.p
    acados_model.name = model.name
    
    # Füge den nichtlinearen Constraint hinzu
    # acados_model.con_h_expr = constraint.expr
    
    return acados_model, constraint
