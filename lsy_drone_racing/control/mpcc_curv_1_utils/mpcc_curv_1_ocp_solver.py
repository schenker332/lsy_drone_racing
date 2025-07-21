
from __future__ import annotations
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from lsy_drone_racing.control.mpcc_curv_1_utils.mpcc_curv_1_quadrotor_ode_model import export_quadrotor_ode_model
from lsy_drone_racing.control.helper.costfunction import contour_and_lag_error
from casadi import DM, sum1, exp


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

    p = ocp.model.p
    u = ocp.model.u


    ### ========================================= ###
    ### ============ Cost function ============== ###
    ### ========================================= ###
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    e_c, e_l= contour_and_lag_error(ocp.model)

    q_c = 90 # contour error weight
    q_l = 60 # lag error weight
    q_c_gauss = p[6]  

    # Inputs
    q_u_vec = DM([0.06, 0.055, 0.055, 0.055 ])  # Gewichtung für df_cmd, dr_cmd, dp_cmd, dy_cmd, dv_theta_cmd
    control_cost = q_u_vec[0] * u[0]**2 + q_u_vec[1] * u[1]**2 + q_u_vec[2] * u[2]**2 + q_u_vec[3] * u[3]**2 

    ocp.model.cost_expr_ext_cost    = q_c_gauss * e_c**2 + q_l * e_l**2   + control_cost    
    ocp.model.cost_expr_ext_cost_e  = 0


    ### ========================================= ###
    ### ============ Constraints ================ ###
    ### ========================================= ###
    ocp.constraints.x0 = np.zeros(nx)
    ocp.parameter_values = np.zeros(np_)

    ocp.constraints.lbx = np.array([0.1, 0.1, -1, -0.9, -1.57])
    ocp.constraints.ubx = np.array([0.55, 0.55, 1, 0.9, 1.57])
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

   

    ### ========================================== ###
    ### ============ Dynamics ==================== ###
    ### ========================================== ###
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM" # FULL_CONDENSING_HPIPM
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # GAUSqcS_NEWTON
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI
    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.nlp_solver_max_iter = 50 

    solver = AcadosOcpSolver(ocp, json_file=ocp.json_file, verbose=verbose)
    return solver, ocp
