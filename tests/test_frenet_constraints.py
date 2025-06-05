#!/usr/bin/env python3
"""
Test für die Frenet-basierten Constraints im MPC-Controller.

Dieser Test zeigt, wie die Frenet-basierten Constraints in den MPC-Controller 
integriert werden können, ohne die Systemdynamik zu ändern.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Füge den Projektpfad zum Pythonpfad hinzu
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Importiere die notwendigen Module
from lsy_drone_racing.control.create_ocp_solver import create_ocp_solver
from lsy_drone_racing.control.frenet_constraints import (
    add_frenet_constraint_to_ocp, 
    get_frenet_coordinates_from_state
)
from lsy_drone_racing.control.export_quadrotor_ode_model import export_quadrotor_ode_model

def test_frenet_constraints():
    """
    Testet die Integration von Frenet-basierten Constraints in den MPC-Controller.
    """
    print("Teste Frenet-basierte Constraints für den MPC-Controller...")
    
    # Erstelle das Modell und die Constraints
    model, constraint = export_quadrotor_ode_model()
    
    # Demonstriere, wie der bestehende create_ocp_solver mit Frenet-Constraints erweitert werden könnte
    
    def create_ocp_solver_with_frenet(Tf: float, N: int, verbose: bool = False):
        """
        Erweitert den create_ocp_solver um Frenet-basierte Constraints.
        Dies ist nur ein Beispiel und ändert nicht die tatsächliche Implementierung.
        """
        # Erstelle den Standard-Solver
        solver, ocp = create_ocp_solver(Tf, N, verbose)
        
        # Hier würde man den Frenet-basierten Constraint hinzufügen
        print("Füge Frenet-basierten Constraint zum OCP hinzu...")
        # ocp = add_frenet_constraint_to_ocp(ocp, constraint, model)
        
        # In einer tatsächlichen Implementierung würde man hier den aktualisierten Solver zurückgeben
        # solver = AcadosOcpSolver(ocp, json_file=ocp.json_file, verbose=verbose)
        
        return solver, ocp
    
    # Test mit einigen Beispielpunkten
    print("\nTeste die Berechnung von Frenet-Koordinaten aus Zuständen...")
    
    # Beispiel-Zustände und Referenzpunkte
    state = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ref_point = np.array([0.0, 0.0, 0.0])
    next_ref_point = np.array([2.0, 2.0, 2.0])
    
    # Berechne Frenet-Koordinaten
    s, n, h = get_frenet_coordinates_from_state(state, ref_point, next_ref_point)
    
    print(f"Zustand (Position): ({state[0]}, {state[1]}, {state[2]})")
    print(f"Referenzpunkt: ({ref_point[0]}, {ref_point[1]}, {ref_point[2]})")
    print(f"Nächster Referenzpunkt: ({next_ref_point[0]}, {next_ref_point[1]}, {next_ref_point[2]})")
    print(f"Frenet-Koordinaten: s = {s:.2f}, n = {n:.2f}, h = {h:.2f}")
    
    # Visualisiere die Ergebnisse
    visualize_frenet_example(state, ref_point, next_ref_point, s, n, h)
    
    print("\nDieser Test zeigt, wie die Frenet-basierten Constraints in den MPC-Controller integriert werden könnten.")
    print("Um die tatsächliche Integration durchzuführen, müsste create_ocp_solver.py angepasst werden.")

def visualize_frenet_example(
    state: np.ndarray,
    ref_point: np.ndarray,
    next_ref_point: np.ndarray,
    s: float,
    n: float,
    h: float
):
    """
    Visualisiert ein Beispiel für Frenet-Koordinaten.
    
    Args:
        state: Der Zustandsvektor mit (px, py, pz, ...)
        ref_point: Aktueller Referenzpunkt [x, y, z]
        next_ref_point: Nächster Referenzpunkt [x, y, z]
        s, n, h: Die berechneten Frenet-Koordinaten
    """
    from lsy_drone_racing.control.frenet_adapter import visualize_current_situation
    
    # Extrahiere die Position aus dem Zustand
    position = (state[0], state[1], state[2])
    
    # Verwende die Visualisierungsfunktion aus frenet_adapter
    visualize_current_situation(
        drone_position=position,
        ref_point=tuple(ref_point),
        next_ref_point=tuple(next_ref_point),
        tube_radius=0.3  # Beispiel-Radius
    )

if __name__ == "__main__":
    test_frenet_constraints()
