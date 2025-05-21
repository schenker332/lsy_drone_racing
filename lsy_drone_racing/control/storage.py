#===== create ocp solver


# # Get Dimensions
# nx = model.x.rows()
# nu = model.u.rows()
# ny = nx + nu + 3
# ny_e = nx + 3


# W_gate = 1000
# ocp.cost.W = scipy.linalg.block_diag(Q, R, np.diag([W_gate, W_gate, W_gate]))
# ocp.cost.W_e = scipy.linalg.block_diag(Q_e, np.diag([W_gate, W_gate, W_gate]))

# Vx = np.zeros((ny, nx))     # 21x14   
# Vx[:nx, :] = np.eye(nx)         # oben
# Vx[nx + nu : nx + nu + 3, 0:3] = np.eye(3)  # unten

# Vx_e = np.zeros((ny_e, nx))     
# Vx_e[:nx, :nx] = np.eye(nx)
# Vx_e[nx : nx + 3, 0:3] = np.eye(3)  # unten

# Vu = np.zeros((ny, nu))             #  21x4
# Vu[nx : nx + nu, :] = np.eye(nu)    # mitte


#====== compute control

# gate_id  = obs["target_gate"]
# gate_pos = obs["gates_pos"][gate_id]      # echte Koordinaten des Gates
# seen     = obs["gates_visited"][gate_id]  # ob schon gesehen

# for j in range(self.N):
#     # Position aus Trajektorie
#     x_ref = self.x_des[i + j]
#     y_ref = self.y_des[i + j]
#     z_ref = self.z_des[i + j]

#     # Dynamisches Umschalten des Gate-Ziels
#     if seen:
#         gx, gy, gz = gate_pos
#     else:
#         gx, gy, gz = x_ref, y_ref, z_ref  # noch nicht gesehen → nutze geplante Koordinate

#     yref = np.array([
#         x_ref, y_ref, z_ref,
#         0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0,
#         0.35, 0.35,
#         0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0,0.0,
#         gx, gy, gz  # Gate-Slot dynamisch
#     ])
#     self.acados_ocp_solver.set(j, "yref", yref)


# if seen:
#     gx, gy, gz = gate_pos
# else:
#     gx, gy, gz = self.x_des[i + self.N], self.y_des[i + self.N], self.z_des[i + self.N]

# yref_N = np.array([
#     self.x_des[i + self.N],
#     self.y_des[i + self.N],
#     self.z_des[i + self.N],
#     0.0, 0.0, 0.0,
#     0.0, 0.0, 0.0,
#     0.35, 0.35,
#     0.0, 0.0, 0.0,
#     gx, gy, gz
# ])
# self.acados_ocp_solver.set(self.N, "yref", yref_N)




    # model.p = MX.sym("dummy_params", 12)  # Placeholder, actual values set in create_ocp_solver
#### ====== compute control 2 =====


#         # ====== set constraint parameters (R and g) ======
#         gate_id = obs["target_gate"]
#         if obs["gates_visited"][gate_id]:
#             gate_R = R.from_quat(obs["gates_quat"][gate_id]).as_matrix()
#             gate_pos = obs["gates_pos"][gate_id]
#             p_vec = np.hstack([gate_R.flatten(), gate_pos])
#             for k in range(self.N + 1):
#                 self.acados_ocp_solver.set(k, "p", p_vec)



# # # ====== solve OCP ======


# from __future__ import annotations  # Python 3.10 type hints
# from typing import TYPE_CHECKING
# import numpy as np
# import scipy
# from casadi import vertcat, MX, reshape
# from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
# from lsy_drone_racing.control.export_quadrotor_ode_model import export_quadrotor_ode_model



# def create_ocp_solver(
#     Tf: float, N: int, verbose: bool = False
# ) -> tuple[AcadosOcpSolver, AcadosOcp]:
#     """Creates an acados Optimal Control Problem and Solver."""
#     ocp = AcadosOcp()

#     # set model
#     model = export_quadrotor_ode_model()
#     ocp.model = model





#     # Get Dimensions
#     nx = model.x.rows()
#     nu = model.u.rows()
#     ny = nx + nu 
#     ny_e = nx 

#     # Set dimensions
#     ocp.solver_options.N_horizon = N

#     # Cost Type
#     ocp.cost.cost_type = "LINEAR_LS"
#     ocp.cost.cost_type_e = "LINEAR_LS"
#     Q = np.diag([10.0, 10.0, 10.0, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01])
#     R = np.diag([0.01, 0.01, 0.01, 0.01])
#     Q_e = Q.copy()




#     ocp.cost.W = scipy.linalg.block_diag(Q, R)
#     ocp.cost.W_e = Q_e

#     Vx = np.zeros((ny, nx))
#     Vx[:nx, :] = np.eye(nx)  # Only select position states   18x14  I= 14x14 (oben)

#     Vu = np.zeros((ny, nu))
#     Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions   18x4  I= 4x4 (unten)

#     Vx_e = np.zeros((ny_e, nx))
#     Vx_e[:nx, :nx] = np.eye(nx)  # Only select position states  14x14  I= 14x14 

#     ocp.cost.Vx = Vx
#     ocp.cost.Vu = Vu
#     ocp.cost.Vx_e = Vx_e




#     # Set initial references (we will overwrite these later on to make the controller track the traj.)
#     # ocp.cost.yref = np.zeros((ny, ))
#     # ocp.cost.yref_e = np.zeros((ny_e, ))
#     ocp.cost.yref = np.array(
#         [1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#     )

#     ocp.cost.yref_e = np.array(
#         [1.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0]
#     )



#     #==== constraints ====

#     R_sym = MX.sym("R_gate", 9)     # 3×3 Rotation → 9-Vektor
#     g_sym = MX.sym("g_gate", 3)     # Gate-Zentrum
#     R_mat = reshape(R_sym, 3, 3)

#     p_w   = model.x[0:3]             # (px,py,pz)
#     p_g   = R_mat @ (p_w - g_sym)   # (u,v,w)
#     u, v, w = p_g[0], p_g[1], p_g[2]




#     gate = 0.30/ 2  # Gate-Breite
#     d2      = 0.30 / 2  # halbe Dicke der Wand (z-Richtung)
#     frame = 2 



#     def box_6(u_min, u_max, v_min, v_max):
#         return vertcat(
#             u   - u_max,      # u ≤ u_max
#         -u   + u_min,      # u ≥ u_min
#             v   - v_max,      # v ≤ v_max
#         -v   + v_min,      # v ≥ v_min
#             w   - d2,         # w ≤ +d2
#         -w   + (-d2)       # w ≥ -d2
#         )



#     # 5) Vier Klötze zusammensetzen ---------------------------------
#     con_h = vertcat(
#         # links
#         box_6(-frame, -gate, -frame, frame),
#         # rechts
#         box_6( gate,  frame, -frame, frame),
#         # oben
#         box_6(-gate, gate,  gate, frame),
#         # unten
#         box_6(-gate, gate, -frame, -gate),
#     )

 



#     # 6) ans Modell hängen ------------------------------------------
#     model.con_h_expr = con_h                # 24×1
#     model.p          = vertcat(R_sym, g_sym)  # 12 Parameter

#     # Initialisiere Parameter mit sinnvoller Gate-Konfiguration
#     gate_R0 = np.eye(3)  # Gate-Rotation = Identitätsmatrix (d.h. kein gedrehter Rahmen)
#     gate_pos0 = np.array([0.5, 0.0, 1.0])  # Gate-Zentrum liegt vor der Drohne auf 1 m Höhe
#     ocp.parameter_values = np.hstack([gate_R0.flatten(), gate_pos0])

#     # Set constraint bounds for model.con_h_expr
# # In create_ocp_solver.py
#     ocp.constraints.lh = -1e10 * np.ones(24)
#     ocp.constraints.uh = np.zeros(24)




#     # hard constraints state
#     ocp.constraints.lbx = np.array([0.1, 0.1, -1.57, -1.57, -1.57])
#     ocp.constraints.ubx = np.array([0.55, 0.55, 1.57, 1.57, 1.57])
#     ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

#     # hard constraints input
#     # ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0. -10.0])
#     # ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0])
#     # ocp.constraints.idxbu = np.array([0, 1, 2, 3])






#     # === Gate constraint evaluation for (u,v,w) ===
#     def check_gate_constraints(u_val: float, v_val: float, w_val: float) -> np.ndarray:
#         """Evaluate gate constraints for given transformed position (u, v, w).
#         Returns: np.ndarray of 24 booleans. True=satisfied, False=violated.
#         """
#         import numpy as np
#         d2 = 0.30 / 2
#         gate = 0.30 / 2
#         frame = 2

#         def box_6_eval(u, v, w, u_min, u_max, v_min, v_max):
#             return np.array([
#                 u - u_max <= 0,
#                 -u + u_min <= 0,
#                 v - v_max <= 0,
#                 -v + v_min <= 0,
#                 w - d2 <= 0,
#                 -w + (-d2) <= 0
#             ])

#         constraints = np.concatenate([
#             box_6_eval(u_val, v_val, w_val, -frame, -gate, -frame, frame),   # links
#             box_6_eval(u_val, v_val, w_val,  gate,  frame, -frame, frame),   # rechts
#             box_6_eval(u_val, v_val, w_val, -gate,  gate,  gate,  frame),    # oben
#             box_6_eval(u_val, v_val, w_val, -gate,  gate, -frame, -gate),    # unten
#         ])
#         return constraints
        
#     satisfied = check_gate_constraints(u, v, w)

#     from casadi import Function

#     # Create CasADi function to evaluate con_h
#     con_fun = Function("con_h_fun", [model.x, model.p], [con_h])

#     # Provide test values
#     x_test = np.zeros((nx,))
#     p_test = np.zeros((12,))

#     # Evaluate constraint function numerically
#     con_vals = con_fun(x_test, p_test).full().flatten()

#     # Evaluate if constraints are violated
#     satisfied = con_vals <= 0
#     print("All constraints satisfied?", satisfied.all())
#     print("Which failed?", np.where(~satisfied)[0])




#     # We have to set x0 even though we will overwrite it later on.
#     ocp.constraints.x0 = np.zeros((nx))

    




#     # Solver Options
#     ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
#     ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
#     ocp.solver_options.integrator_type = "ERK"
#     ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI
#     ocp.solver_options.tol = 1e-5

#     ocp.solver_options.qp_solver_cond_N = N
#     ocp.solver_options.qp_solver_warm_start = 1

#     ocp.solver_options.qp_solver_iter_max = 20
#     ocp.solver_options.nlp_solver_max_iter = 50

#     # set prediction horizon
#     ocp.solver_options.tf = Tf


#     # === Solver erzeugen ===
#     acados_ocp_solver = AcadosOcpSolver(ocp, json_file="lsy_example_mpc.json", verbose=verbose)



#     return acados_ocp_solver, ocp
