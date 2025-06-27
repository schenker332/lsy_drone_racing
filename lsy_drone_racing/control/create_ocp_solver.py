
from __future__ import annotations
import numpy as np
import scipy
from acados_template import AcadosOcp, AcadosOcpSolver
from lsy_drone_racing.control.export_quadrotor_ode_model import export_quadrotor_ode_model
from lsy_drone_racing.control.helper.costfunction import create_tracking_cost_function
from casadi import sumsqr


def create_ocp_solver(Tf: float, N: int, verbose: bool = False) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Create an optimal control problem solver for the quadrotor MPC.
    
    Args:
        Tf: Time horizon in seconds.
        N: Number of discretization steps.
        verbose: Whether to print solver information.
        
    Returns:
        tuple: (solver, ocp) containing the acados solver and OCP object.
    """
    # Create acados OCP object
    ocp = AcadosOcp()
    
    # Set up the dynamic model
    ocp.model = export_quadrotor_ode_model()
    ocp.json_file = f"{ocp.model.name}.json"




    # Get dimensions 
    nx = ocp.model.x.size()[0]  # State dimension
    nu = ocp.model.u.size()[0]  # Control input dimension
    ny = nx + nu               # Output dimension (states + controls)
    ny_e = nx                  # Terminal output dimension

    # Set prediction horizon
    ocp.solver_options.N_horizon = N

    ### ==================== old costfunction ==================== ###

    # # Define cost weights
    # # State weights: high weights for position (first 3), lower for velocities, etc.
    # Q = np.diag([10.0] * 3 + [0.01] * 3 + [0.1] * 3 + [0.01] * 5)
    # # Control input weights
    # R = np.diag([0.01] * nu)

    # # Set cost function type to linear least-squares
    # ocp.cost.cost_type = "LINEAR_LS"
    # ocp.cost.cost_type_e = "LINEAR_LS"
    
    # # Combine state and input weights
    # ocp.cost.W = scipy.linalg.block_diag(Q, R)
    # ocp.cost.W_e = Q.copy()  # Terminal state weights

    # # Define output matrices for cost function
    # # State selection matrix
    # Vx = np.zeros((ny, nx))
    # Vx[:nx, :] = np.eye(nx)
    # # Control input selection matrix
    # Vu = np.zeros((ny, nu))
    # Vu[nx:, :] = np.eye(nu)
    # ocp.cost.Vx = Vx
    # ocp.cost.Vu = Vu  

    # # Terminal state selection matrix
    # Vx_e = np.eye(nx)
    # ocp.cost.Vx_e = Vx_e

    # # Initialize reference to zero
    # ocp.cost.yref = np.zeros(ny)
    # ocp.cost.yref_e = np.zeros(ny_e)
    ### ============================================================ ###



    ### ==================== new costfunction ==================== ###
    # Erzeuge Kostenfunktion
    cost_y_expr, cost_y_expr_e= create_tracking_cost_function(ocp.model)
    
    # Get cost dimensions from the expressions

    
    # Set up nonlinear least squares cost
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    
    # Set weights for the cost function
    q_c = 20 # Weight for contour error
    q_l = 5  # Weight for lag error
    q_u = 0.04  # Weight for control inputs
    mu = 5

    # Set the cost expressions
    u = ocp.model.u
    x = ocp.model.x

    # Definiere Steuerungskosten für CasADi-Objekte korrekt

    control_cost = q_u * sumsqr(u)

    # In create_ocp_solver.py
    ocp.model.cost_expr_ext_cost = q_c * cost_y_expr[0]**2 + q_l * cost_y_expr[1]**2 +  control_cost - mu * x[14]  
    ocp.model.cost_expr_ext_cost_e = q_c * cost_y_expr_e[0]**2 + q_l * cost_y_expr_e[1]**2 - mu * x[14]
    ### =========================================================== ###
    






    # Set initial state constraint to zero (will be updated at runtime)
    ocp.constraints.x0 = np.zeros(nx)
    ocp.parameter_values = np.zeros(6)

    ocp.constraints.lbx = np.array([0.1, 0.1, -1.57, -1.57, -1.57])
    ocp.constraints.ubx = np.array([0.55, 0.55, 1.57, 1.57, 1.57])
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

    # Configure solver settings
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"  
    ocp.solver_options.nlp_solver_type = "SQP" 
    ocp.solver_options.tf = Tf                  
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # Create solver using the model name for the JSON file
    solver = AcadosOcpSolver(ocp, json_file=ocp.json_file, verbose=verbose)
    return solver, ocp
