from __future__ import annotations

from acados_template import AcadosModel
from casadi import SX, cos, sin, vertcat


def export_quadrotor_ode_model() -> AcadosModel:
    """Creates and exports a quadrotor ODE model for use with acados.

    This function defines the quadrotor dynamics model including position,
    velocity, orientation, thrust, and control inputs for the MPC controller.
    The model uses a state-space representation with states, controls, and parameters
    that are essential for the Optimal Control Problem (OCP).

    Returns:
        AcadosModel: An AcadosModel object containing the system dynamics and properties.
    """
    model_name = "lsy_example_mpc"

    # Physical parameters of the quadrotor
    GRAVITY = 9.81  # Gravitational acceleration in m/s^2
    # Empirically identified parameters for the drone's dynamics
    params_pitch_rate = [-6.003842038081178, 6.213752925707588]
    params_roll_rate = [-3.960889336015948, 4.078293254657104]
    params_yaw_rate = [-0.005347588299390372, 0.0]
    params_acc = [20.907574256269616, 3.653687545690674]

    # --- Define State Variables ---
    # These variables represent the internal state of the drone at any time.
    px = SX.sym("px")  # Position x
    py = SX.sym("py")  # Position y
    pz = SX.sym("pz")  # Position z
    vx = SX.sym("vx")  # Velocity x
    vy = SX.sym("vy")  # Velocity y
    vz = SX.sym("vz")  # Velocity z
    roll = SX.sym("r")  # Roll angle
    pitch = SX.sym("p")  # Pitch angle
    yaw = SX.sym("y")  # Yaw angle
    f_collective = SX.sym("f_collective")  # Collective thrust
    f_collective_cmd = SX.sym("f_collective_cmd")  # Commanded collective thrust
    r_cmd = SX.sym("r_cmd")  # Commanded roll rate
    p_cmd = SX.sym("p_cmd")  # Commanded pitch rate
    y_cmd = SX.sym("y_cmd")  # Commanded yaw rate


    # --- Define Control Inputs ---
    # These are the variables that the MPC solver can manipulate.
    df_cmd = SX.sym("df_cmd")  # Change in commanded collective thrust
    dr_cmd = SX.sym("dr_cmd")  # Change in commanded roll rate
    dp_cmd = SX.sym("dp_cmd")  # Change in commanded pitch rate
    dy_cmd = SX.sym("dy_cmd")  # Change in commanded yaw rate


    # --- Define Online Parameters ---
    # These parameters are updated at each MPC step to provide new references or weights.
    x_ref = SX.sym("x_ref")  # Current x-coordinate of the reference trajectory
    y_ref = SX.sym("y_ref")  # Current y-coordinate of the reference trajectory
    z_ref = SX.sym("z_ref")  # Current z-coordinate of the reference trajectory
    x_ref_next = SX.sym("x_ref_next")  # Next x-coordinate for tangent calculation
    y_ref_next = SX.sym("y_ref_next")  # Next y-coordinate for tangent calculation
    z_ref_next = SX.sym("z_ref_next")  # Next z-coordinate for tangent calculation
    q_c_gauss = SX.sym("q_c_gauss")  # Weight for the contouring error cost

    # --- State and Control Vectors ---
    states = vertcat(
        px, py, pz, # 0 , 1, 2
        vx, vy, vz, # 3, 4, 5
        roll, pitch, yaw, # 6, 7, 8
        f_collective, f_collective_cmd, r_cmd, # 9, 10, 11
        p_cmd, y_cmd # 12, 13


    )
    controls = vertcat(df_cmd, dr_cmd, dp_cmd, # 0, 1, 2
                       dy_cmd) # 3
    
    p = vertcat(x_ref, y_ref, z_ref, # 0, 1, 2
                x_ref_next, y_ref_next, z_ref_next, # 3, 4, 5
                q_c_gauss # 6
                ) 
    # System dynamics
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
        10.0 * (f_collective_cmd - f_collective),  # Thrust dynamics
        df_cmd,
        dr_cmd,
        dp_cmd,
        dy_cmd
    )

    # --- Create the AcadosModel ---
    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.x = states
    model.u = controls
    model.p = p
    model.name = model_name

    return model
