from __future__ import annotations

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from casadi import DM

from lsy_drone_racing.control.export_quadrotor_ode_model import export_quadrotor_ode_model
from lsy_drone_racing.control.helper.costfunction import contour_and_lag_error, get_min_distance_to_trajectory
from casadi import DM, sum1, exp


def create_ocp_solver(Tf: float, N: int, verbose: bool = False) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Create an Optimal Control Problem (OCP) solver for the quadrotor MPC.

    This function sets up the OCP with the quadrotor's dynamic model,
    cost function, and constraints. It then creates an Acados solver
    instance that can be used for real-time control.

    Args:
        Tf: The time horizon in seconds for the MPC.
        N: The number of discretization steps over the time horizon.
        verbose: If True, prints solver information during creation.

    Returns:
        A tuple containing the AcadosOcpSolver instance and the AcadosOcp object.
    """
    ocp = AcadosOcp()

    # Set up the dynamic model from the exported model file
    ocp.model = export_quadrotor_ode_model()
    ocp.json_file = f"{ocp.model.name}.json"

    # Set the prediction horizon
    ocp.solver_options.N_horizon = N

    # Get dimensions of the model
    nx = ocp.model.x.size()[0]  # State dimension
    np_ = ocp.model.p.size()[0]  # Parameter dimension

    # Extract symbolic variables for easier access
    p = ocp.model.p
    u = ocp.model.u

    # --- Cost Function ---
    # The cost is defined as an external cost, allowing for a more complex function
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"  # Terminal cost

    # Decompose the tracking error into contouring and lag errors
    e_c, e_l = contour_and_lag_error(ocp.model)

    # --- Cost Function Weights ---

    q_c = 90 # contour error weight
    q_l = 60.0  # Weight for the lag error (progress along the path)

    # The contouring error weight is set as an online parameter in the model/controller
    theta_spec_cont_weight = p[6]

    # --- Control Input Cost ---
    # Weights for the control inputs to penalize excessive commands
    q_u_vec = DM([0.06, 0.055, 0.055, 0.055])  # df_cmd, dr_cmd, dp_cmd, dy_cmd, dv_theta_cmd
    control_cost = (
        q_u_vec[0] * u[0] ** 2
        + q_u_vec[1] * u[1] ** 2
        + q_u_vec[2] * u[2] ** 2
        + q_u_vec[3] * u[3] ** 2
    )

     # --- Total Cost Expression ---
    # The total cost is a sum of contouring error, lag error, and control input costs.
    # A quadratic cost on v_theta helps stabilize the progress along the trajectory.
    ### LEARNING: Progress v_theta quadratic brings huge stabilisation
    ocp.model.cost_expr_ext_cost    = theta_spec_cont_weight * e_c**2 + q_l * e_l**2  + control_cost      
    ocp.model.cost_expr_ext_cost_e  = 0 


    ### ========================================= ###
    ### ============ Constraints ================ ###
    ### ========================================= ###
    ocp.constraints.x0 = np.zeros(nx)
    ocp.parameter_values = np.zeros(np_)


    ocp.constraints.lbx = np.array([0.1, 0.1, -1, -0.9, -1.57])
    ocp.constraints.ubx = np.array([0.55, 0.55, 1, 0.9, 1.57])
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])  # Indices of constrained states


    ### ========================================== ###
    ### ============ Solver options ============== ###
    ### ========================================== ###
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM" # FULL_CONDENSING_HPIPM
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.nlp_solver_max_iter = 50

    # Create the solver instance
    solver = AcadosOcpSolver(ocp, json_file=ocp.json_file, verbose=verbose)
    return solver, ocp
