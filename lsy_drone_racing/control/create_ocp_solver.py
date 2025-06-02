
from __future__ import annotations
import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from lsy_drone_racing.control.export_quadrotor_ode_model import export_quadrotor_ode_model
from casadi import MX, cos, sin, vertcat


def create_ocp_solver(Tf: float, N: int, verbose: bool = False) -> tuple[AcadosOcpSolver, AcadosOcp]:
    # OCP erstellen
    ocp = AcadosOcp()
    
    # Modell und Constraint laden
    model, constraint = export_quadrotor_ode_model()
    ocp.model = model
    
    # Wichtig: Acados-ocp-json-Datei mit Modellnamen benennen
    ocp.json_file = f"{model.name}.json"

    # Dimensionen ermitteln - size()[0] ist konsistenter als shape[0]
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx

    ocp.solver_options.N_horizon = N

    Q = np.diag([10.0] * 3 + [0.01] * 3 + [0.1] * 3 + [0.01] * 5)
    R = np.diag([0.01] * nu)

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q.copy()

    Vx = np.zeros((ny, nx))
    Vx[:nx, :] = np.eye(nx)
    Vu = np.zeros((ny, nu))
    Vu[nx:, :] = np.eye(nu)
    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu

    Vx_e = np.eye(nx)
    ocp.cost.Vx_e = Vx_e

    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # Initial state
    ocp.constraints.x0 = np.zeros(nx)

    # Nichtlinearer Tube-Constraint
    ocp.parameter_values = np.zeros(3)  # x_ref, y_ref, z_ref für den Constraint
    tube_radius_sq = constraint.tube_radius**2
    
    # Dimension des Constraints berücksichtigen
    nh = getattr(constraint, 'shape', 1)
    ocp.constraints.lh = np.array([-1.0e9] * nh)  # Sehr niedriger Wert statt -np.inf
    ocp.constraints.uh = np.array([tube_radius_sq] * nh)  # Obere Schranke ist radius^2

    # State and input bounds (hard constraints)
    ocp.constraints.lbx = np.array([0.1, 0.1, -1.57, -1.57, -1.57])
    ocp.constraints.ubx = np.array([0.55, 0.55, 1.57, 1.57, 1.57])
    ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

    # Solver settings
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # Verwende den Modellnamen für die JSON-Datei
    solver = AcadosOcpSolver(ocp, json_file=ocp.json_file, verbose=verbose)
    return solver, ocp
