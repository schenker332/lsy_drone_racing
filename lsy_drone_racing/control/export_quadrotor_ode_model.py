from __future__ import annotations
from acados_template import AcadosModel
from casadi import  cos, sin, vertcat, SX, sumsqr
from lsy_drone_racing.control.helper.costfunction import create_tracking_cost_function

def export_quadrotor_ode_model():
    """Create and export a quadrotor ODE model for use with acados.
    
    This function defines the quadrotor dynamics model including position,
    velocity, orientation, thrust and control inputs for the MPC controller.
    
    Returns:
        AcadosModel: Model object containing the system dynamics and properties.
    """
    model_name = "lsy_example_mpc"

    # Physical parameters
    GRAVITY = 9.81
    params_pitch_rate = [-6.003842038081178, 6.213752925707588]
    params_roll_rate = [-3.960889336015948, 4.078293254657104]
    params_yaw_rate = [-0.005347588299390372, 0.0]
    params_acc = [20.907574256269616, 3.653687545690674]
    
    # Define states
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

    # Define additional states for MPCC progress penalisation
    theta = SX.sym("theta")  
    v_theta = SX.sym("v_theta")  
    dv_theta_cmd = SX.sym("dv_theta_cmd") 
    
    # Define inputs
    df_cmd = SX.sym("df_cmd")
    dr_cmd = SX.sym("dr_cmd")
    dp_cmd = SX.sym("dp_cmd")
    dy_cmd = SX.sym("dy_cmd")

    # Reference points (for trajectory tracking)
    x_ref = SX.sym("x_ref")
    y_ref = SX.sym("y_ref")
    z_ref = SX.sym("z_ref")
    x_ref_next = SX.sym("x_ref_next")
    y_ref_next = SX.sym("y_ref_next")
    z_ref_next = SX.sym("z_ref_next")
    


    
    # State and input vectors
    states = vertcat(
        px, py, pz, vx, vy, vz, roll, pitch, yaw,
        f_collective, f_collective_cmd, r_cmd, p_cmd, y_cmd,
        theta,
        v_theta
    )
    #control inputs to the drone
    controls = vertcat(df_cmd, dr_cmd, dp_cmd, dy_cmd, dv_theta_cmd)
    p = vertcat(x_ref, y_ref, z_ref, x_ref_next, y_ref_next, z_ref_next)

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
        dy_cmd,
        v_theta,        # Dynamics for theta: θ̇ = v_theta
        dv_theta_cmd    # Dynamics for v_theta: v̇_theta = dv_theta_cmd
    )

    ### Added for custom cost function ###
    # Set weights for the cost function
    q_c = 60000 # Weight for contour error (1100)
    q_l = 300  # Weight for lag error (8000)
    q_u = 0.0002  # Weight for control inputs
    q_vtheta = 1 # Penalty on high progress velocity
    mu = 0.60   # Weight for progress reward (mu)


    # ## experimental
    # q_att_cmd = 25.0 #Add a new, significant penalty on large roll/pitch commands.
    #  # Set the cost expressions
    # # Definiere Steuerungskosten für CasADi-Objekte 
    # attitude_cmd_cost = q_att_cmd * (r_cmd**2 + p_cmd**2)



     # Set the cost expressions
    # Definiere Steuerungskosten für CasADi-Objekte korrekt

    control_cost = q_u * sumsqr(controls)

    
    # Create the AcadosModel
    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.x = states
    model.u = controls
    model.p = p
    model.name = model_name

    # ### ==================== new costfunction ==================== ###
    # Erzeuge Kostenfunktion
    e_cont, e_lag, e_cont_e, e_lag_e = create_tracking_cost_function(model)

     # In create_ocp_solver.py  definition of our stage cost function
    model.cost_expr_ext_cost = q_c * e_cont**2 + q_l * e_lag**2 - mu * v_theta + q_vtheta * v_theta**2 + control_cost # experimental (did not really help)+ attitude_cmd_cost
    # Definition of the stage cost function for the terminal stage
    model.cost_expr_ext_cost_e = q_c * e_cont**2 + q_l * e_lag**2

    return model
