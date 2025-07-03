
from __future__ import annotations
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from lsy_drone_racing.control.export_quadrotor_ode_model import export_quadrotor_ode_model
from lsy_drone_racing.control.helper.costfunction import contour_and_lag_error, get_min_distance_to_trajectory
from casadi import DM, sum1


def create_ocp_solver(Tf: float, N: int, verbose: bool = False) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Create an optimal control problem solver for the quadrotor MPC.
    
    Args:
        Tf: Time horizon in seconds.
        N: Number of discretization steps.
        verbose: Whether to print solver information.
        
    Returns:
        tuple: (solver, ocp) containing the acados solver and OCP object.
    """

    ocp = AcadosOcp()
    
    # Set up the dynamic model
    ocp.model = export_quadrotor_ode_model()
    ocp.json_file = f"{ocp.model.name}.json"

    ocp.solver_options.N_horizon = N 

    # Get dimensions 
    nx = ocp.model.x.size()[0]  # State dimension
    np_ = ocp.model.p.size()[0]  # Parameter dimension
    

    e_c, e_l= contour_and_lag_error(ocp.model)
    min_distance = get_min_distance_to_trajectory(ocp.model)  # Get minimum distance to trajectory


    p = ocp.model.p
    u = ocp.model.u
    x = ocp.model.x

    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    # Parameters Niclas
    # q_c = 60 # contour error weight
    # q_l = 40 # lag error weight
    # mu = 0.002 # progress weight


    
    ### LEARNING: Similar as for gate penalties, too high contouing error can caouse frek instability incidents
    q_c = 70 # contour error weight
    q_l = 60 # lag error weight
    #mu = 0.0008 # progress weight
    ### LEARNING: Progress weight can be really hign and sometimes makes the controller more reliable => 0.6 also worked
    mu = 0.002  # progress 0.0015 
    q_min = p[6]  # gaussian weight
    max_v_theta = 0.16  # maximum progress velocity
    dv_theta_max = 0.35  # maximum progress acceleration


    # Hover Thrust
    MASS    = 0.033       
    GRAVITY = 9.81     
    f_coll = x[9]
    f_coll_cmd = x[10]
    hover_error = f_coll - MASS * GRAVITY
    hover_error_cmd = f_coll_cmd - MASS * GRAVITY


    # Inputs
    ## OLD more conservartive weights from max
    # q_u_vec = DM([0.02, 0.05, 0.05, 0.05, 0.05])

    q_u_vec = DM([0.06, 0.055, 0.055, 0.055, 0.05 ])  # Gewichtung für df_cmd, dr_cmd, dp_cmd, dy_cmd, dv_theta_cmd
    weighted_squares = q_u_vec * (u**2)
    # control_cost = sum1(weighted_squares)
    control_cost = q_u_vec[0] * u[0]**2 + q_u_vec[1] * u[1]**2 + q_u_vec[2] * u[2]**2 + q_u_vec[3] * u[3]**2 + q_u_vec[4] * (u[4]**2)

    # Set cost funnction
    ### LEARNING: Progress v_theta quadratic brings huge stabilisation
    ocp.model.cost_expr_ext_cost    = q_c * e_c**2 + q_l * e_l**2   - mu * x[15]**2   + q_min * min_distance**2   + control_cost     #+ hover_error**2 + hover_error_cmd**2 
    ocp.model.cost_expr_ext_cost_e  = q_c * e_c**2 + q_l * e_l**2   - mu * x[15]**2   + q_min * min_distance**2
	
    ### OPTION: with quadratic cost from v5 Branch
    # ocp.model.cost_expr_ext_cost    = q_c * e_c**2 + q_l * e_l**2   - mu * x[15]**2   + q_min * min_distance**2   + control_cost     #+ hover_error**2 + hover_error_cmd**2 
    # ocp.model.cost_expr_ext_cost_e  = q_c * e_c**2 + q_l * e_l**2   - mu * x[15]**2   + q_min * min_distance**2




    


    ocp.constraints.x0 = np.zeros(nx)
    ocp.parameter_values = np.zeros(np_)

    ### Option: also limit state 14 between 0 and 1 but that should not be needed (from v5 branch)

    ocp.constraints.lbx = np.array([0.1, 0.1, -1, -0.9, -1.57, -max_v_theta])
    ocp.constraints.ubx = np.array([0.55, 0.55, 1, 0.9, 1.57, max_v_theta])
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13, 15])

    # add a limit for the progress acceleration dv_theta_cmd
    
    ocp.constraints.lbu = np.array([-dv_theta_max])
    ocp.constraints.ubu = np.array([dv_theta_max])
    ocp.constraints.idxbu = np.array([4])  # position of dvtheta_cmd in input vector
   
    # Solver Options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM" # FULL_CONDENSING_HPIPM
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # GAUSqcS_NEWTON
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI
    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.nlp_solver_max_iter = 50 
    # ocp.solver_options.tol = 1e-3    
    # ocp.solver_options.nlp_solver_ext_qp_res = 1
    # ocp.solver_options.regularize_method  = "CONVEXIFY"
    # ocp.solver_options.globalization_line_search_use_sufficient_descent = 1

    solver = AcadosOcpSolver(ocp, json_file=ocp.json_file, verbose=verbose)
    return solver, ocp
