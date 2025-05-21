from __future__ import annotations  # Python 3.10 type hints
import numpy as np
import scipy
from acados_template import  AcadosOcp, AcadosOcpSolver
from lsy_drone_racing.control.export_quadrotor_ode_model import export_quadrotor_ode_model



def create_ocp_solver(
    Tf: float, N: int, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # set model
    model = export_quadrotor_ode_model()
    ocp.model = model



    # Get Dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu 
    ny_e = nx 

    # Set dimensions
    ocp.solver_options.N_horizon = N

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    Q = np.diag([10.0, 10.0, 10.0, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01])
    R = np.diag([0.01, 0.01, 0.01, 0.01])

    Q_e = Q.copy()

    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e


    Vx = np.zeros((ny, nx))
    Vx[:nx, :] = np.eye(nx)  # Only select position states   18x14  I= 14x14 (oben)
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions   18x4  I= 4x4 (unten)
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:nx, :nx] = np.eye(nx)  # Only select position states  14x14  I= 14x14 
    ocp.cost.Vx_e = Vx_e








    # Set initial references (we will overwrite these later on to make the controller track the traj.)
    # ocp.cost.yref = np.zeros((ny, ))
    # ocp.cost.yref_e = np.zeros((ny_e, ))
    ocp.cost.yref = np.array(
        [1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    ocp.cost.yref_e = np.array(
        [1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0]
    )

    # Set State Constraints
    ocp.constraints.lbx = np.array([0.1, 0.1, -1.57, -1.57, -1.57])
    ocp.constraints.ubx = np.array([0.55, 0.55, 1.57, 1.57, 1.57])
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

    # Set Input Constraints
    # ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0. -10.0])
    # ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0])
    # ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI
    ocp.solver_options.tol = 1e-5

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="lsy_example_mpc.json", verbose=verbose)

    return acados_ocp_solver, ocp


