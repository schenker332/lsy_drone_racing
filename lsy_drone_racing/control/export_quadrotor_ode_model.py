from __future__ import annotations
from acados_template import AcadosModel
from casadi import  cos, sin, vertcat, SX

def export_quadrotor_ode_model():
    """Create and export a quadrotor ODE model for use with acados.
    
    This function defines the quadrotor dynamics model including position,
    velocity, orientation, thrust and control inputs for the MPC controller.
    
    Returns:
        AcadosModel: Model object containing the system dynamics and properties.
    """
    model_name = "lsy_example_mpc"

    # Physical parameters
    GRAVITY = 9.81 # m/s^2, gravitational acceleration
    params_pitch_rate = [-6.003842038081178, 6.213752925707588]
    params_roll_rate = [-3.960889336015948, 4.078293254657104]
    params_yaw_rate = [-0.005347588299390372, 0.0]
    params_acc = [20.907574256269616, 3.653687545690674]
    
    # state
    px = SX.sym("px") #0
    py = SX.sym("py") #1
    pz = SX.sym("pz") #2
    vx = SX.sym("vx") #3
    vy = SX.sym("vy") #4
    vz = SX.sym("vz") #5
    roll = SX.sym("r") #6
    pitch = SX.sym("p") #7
    yaw = SX.sym("y") #8
    f_collective = SX.sym("f_collective") #9
    f_collective_cmd = SX.sym("f_collective_cmd") #10
    r_cmd = SX.sym("r_cmd") #11
    p_cmd = SX.sym("p_cmd") #12
    y_cmd = SX.sym("y_cmd") #13


    # input
    df_cmd = SX.sym("df_cmd")
    dr_cmd = SX.sym("dr_cmd")
    dp_cmd = SX.sym("dp_cmd")
    dy_cmd = SX.sym("dy_cmd")


    # parameter
    x_ref = SX.sym("x_ref") #0
    y_ref = SX.sym("y_ref") #1
    z_ref = SX.sym("z_ref") #2
    x_ref_next = SX.sym("x_ref_next") #3
    y_ref_next = SX.sym("y_ref_next") #4
    z_ref_next = SX.sym("z_ref_next") #5
    q_c_gauss = SX.sym("q_c_gauss")  # Cost for minimum distance to trajectory #6



    # State and input vectors
    states = vertcat(
        px, py, pz, # 0 , 1, 2
        vx, vy, vz, # 3, 4, 5
        roll, pitch, yaw, # 6, 7, 8
        f_collective, f_collective_cmd, r_cmd, # 9, 10, 11
        p_cmd, y_cmd # 12, 13

    )
    controls = vertcat(df_cmd, dr_cmd, dp_cmd, # 0, 1, 2
                       dy_cmd) # 3, 4
    
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
        10.0 * (f_collective_cmd - f_collective),
        df_cmd,
        dr_cmd,
        dp_cmd,
        dy_cmd
    )



    
    # Create the AcadosModel
    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.x = states
    model.u = controls
    model.p = p
    model.name = model_name

    return model
