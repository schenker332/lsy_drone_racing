#!/usr/bin/env python3
"""
Einfacher Test für das Frenet-Koordinatensystem.

Dieser Test erstellt eine einfache 3D-Referenzlinie und demonstriert die 
Konvertierung zwischen kartesischen und Frenet-Koordinaten mit visuellen Beispielen.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Füge den Projektpfad zum Pythonpfad hinzu, um Importe zu ermöglichen
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Importiere die Frenet-Utilities
from lsy_drone_racing.control.frenet_utils import (
    cartesian_to_frenet_3d,
    frenet_to_cartesian_3d,
    compute_path_length
)

def simple_frenet_test():
    """
    Führt einen einfachen Test der Frenet-Koordinaten durch.
    """
    print("Starte einfachen Frenet-Koordinaten-Test...")
    
    # Erstelle eine einfache 3D-Referenzlinie (eine Helix)
    t = np.linspace(0, 4*np.pi, 100)
    radius = 5.0
    height_scale = 1.0
    
    # Referenzpunkte definieren
    ref_points = np.zeros((len(t), 3))
    ref_points[:, 0] = radius * np.cos(t)  # x-Koordinaten
    ref_points[:, 1] = radius * np.sin(t)  # y-Koordinaten
    ref_points[:, 2] = height_scale * t    # z-Koordinaten (ansteigende Höhe)
    
    # Berechne kumulative Längen für bessere s-Werte
    lengths = compute_path_length(ref_points)
    
    # Einfache Tests mit einigen Punkten
    test_points = [
        # Punkt direkt auf der Referenzlinie
        (ref_points[20, 0], ref_points[20, 1], ref_points[20, 2]),
        # Punkt mit positivem n (außerhalb der Helix)
        (ref_points[40, 0] + 2.0, ref_points[40, 1] + 2.0, ref_points[40, 2]),
        # Punkt mit negativem n (innerhalb der Helix)
        (ref_points[60, 0] - 2.0, ref_points[60, 1] - 2.0, ref_points[60, 2]),
        # Punkt mit positivem h (über der Referenzlinie)
        (ref_points[80, 0], ref_points[80, 1], ref_points[80, 2] + 3.0)
    ]
    
    # Konvertiere Test-Punkte zu Frenet-Koordinaten und zurück
    results = []
    for i, (x, y, z) in enumerate(test_points):
        print(f"\n--- Test Punkt {i+1}: ({x:.2f}, {y:.2f}, {z:.2f}) ---")
        
        # Kartesisch zu Frenet
        s, n, h = cartesian_to_frenet_3d(x, y, z, ref_points)
        print(f"Frenet-Koordinaten: s = {s:.2f}, n = {n:.2f}, h = {h:.2f}")
        
        # Frenet zurück zu Kartesisch
        x_back, y_back, z_back = frenet_to_cartesian_3d(s, n, h, ref_points)
        print(f"Zurück zu Kartesisch: ({x_back:.2f}, {y_back:.2f}, {z_back:.2f})")
        
        # Berechne Fehler
        error = np.sqrt((x - x_back)**2 + (y - y_back)**2 + (z - z_back)**2)
        print(f"Konvertierungsfehler: {error:.6f}")
        
        # Ergebnisse speichern
        results.append((s, n, h, x_back, y_back, z_back))
    
    # Visualisiere die Ergebnisse
    visualize_results(ref_points, test_points, results)

def visualize_results(ref_points, test_points, results):
    """
    Visualisiert die Referenzlinie, Test-Punkte und konvertierte Punkte.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Referenzlinie plotten
    ax.plot(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], 'b-', label='Referenzlinie')
    
    # Original-Testpunkte plotten
    for i, (x, y, z) in enumerate(test_points):
        ax.scatter(x, y, z, c='r', marker='o', s=100, label=f'Testpunkt {i+1}' if i == 0 else '')
    
    # Konvertierte Punkte plotten
    for i, (s, n, h, x, y, z) in enumerate(results):
        ax.scatter(x, y, z, c='g', marker='x', s=100, label=f'Konvertiert {i+1}' if i == 0 else '')
        
        # Verbindungslinie zwischen Original- und konvertiertem Punkt
        x_orig, y_orig, z_orig = test_points[i]
        ax.plot([x_orig, x], [y_orig, y], [z_orig, z], 'k--', alpha=0.3)
    
    # Beschriftungen und Legende
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Frenet-Koordinaten Test')
    ax.legend()
    
    # Achsen gleich skalieren
    max_range = np.array([
        ref_points[:, 0].max() - ref_points[:, 0].min(),
        ref_points[:, 1].max() - ref_points[:, 1].min(),
        ref_points[:, 2].max() - ref_points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (ref_points[:, 0].max() + ref_points[:, 0].min()) * 0.5
    mid_y = (ref_points[:, 1].max() + ref_points[:, 1].min()) * 0.5
    mid_z = (ref_points[:, 2].max() + ref_points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simple_frenet_test()
