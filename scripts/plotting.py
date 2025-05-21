import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
from numpy import array, linspace



def draw_obstacle(ax, center, size=(0.1, 0.1, 2.0), color='red', alpha=0.6):

    # Halbmaße
    dx, dy, dz = np.array(size) / 2

    # Eckpunkte des Quaders
    x = [-dx, dx]
    y = [-dy, dy]
    z = [0 , -1.4]


    vertices = np.array([
        [x[0], y[0], z[0]],
        [x[1], y[0], z[0]],
        [x[1], y[1], z[0]],
        [x[0], y[1], z[0]],
        [x[0], y[0], z[1]],
        [x[1], y[0], z[1]],
        [x[1], y[1], z[1]],
        [x[0], y[1], z[1]],
    ]) + center  # Mittelpunkt verschieben

    # Flächen definieren
    faces = [
        [vertices[i] for i in [0, 1, 2, 3]],  # unten
        [vertices[i] for i in [4, 5, 6, 7]],  # oben
        [vertices[i] for i in [0, 1, 5, 4]],  # Seite x+
        [vertices[i] for i in [2, 3, 7, 6]],  # Seite x-
        [vertices[i] for i in [1, 2, 6, 5]],  # Seite y+
        [vertices[i] for i in [0, 3, 7, 4]],  # Seite y-
    ]

    # Zeichnen
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors='k', linewidths=0.3, alpha=alpha))

    ax.scatter(center[0], center[1], center[2], color='black', s=50)  # Mittelpunkt markieren


def create_box_faces(xmin, xmax, ymin, ymax, zmin, zmax):
    # Erstellt die 6 Flächen eines Quaders (als Listen von 4 Punkten)
    return [
        [[xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin]],  # Boden
        [[xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]],  # Deckel
        [[xmin, ymin, zmin], [xmin, ymin, zmax], [xmin, ymax, zmax], [xmin, ymax, zmin]],  # Seite links
        [[xmax, ymin, zmin], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmax, ymax, zmin]],  # Seite rechts
        [[xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymin, zmax], [xmin, ymin, zmax]],  # vorne
        [[xmin, ymax, zmin], [xmax, ymax, zmin], [xmax, ymax, zmax], [xmin, ymax, zmax]]   # hinten
    ]


def draw_gate(ax, center, quat, inner_size=0.45, outer_size=0.55, depth=0.05, color='blue', alpha=1):
    hs_in = inner_size / 2
    hs_out = outer_size / 2
    dz = depth / 2


    

    # Definiere 4 Quader für die vier Rahmenbereiche (x = Tiefe, y/z = Fläche)
    boxes = [
        [-dz, dz, -hs_out, -hs_in, -hs_out, hs_out],   # links
        [-dz, dz,  hs_in,  hs_out, -hs_out, hs_out],   # rechts
        [-dz, dz, -hs_in,  hs_in,  hs_in,  hs_out],    # oben
        [-dz, dz, -hs_in,  hs_in, -hs_out, -hs_in],    # unten
    ]

    ax.scatter(center[0], center[1], center[2], color='black', s=10)  # Mittelpunkt markieren
    # Rotation vorbereiten
    r_gate = R.from_quat(quat)
    r_fix = R.from_euler('z', 90, degrees=True)
    r_total = r_gate * r_fix

    for x0, x1, y0, y1, z0, z1 in boxes:
        faces = create_box_faces(x0, x1, y0, y1, z0, z1)
        rotated_faces = [] 
        for face in faces:
            face = np.array(face)
            rotated = r_total.apply(face)
            rotated += center
            rotated_faces.append(rotated)
        poly = Poly3DCollection(rotated_faces, facecolors=color, edgecolors='k', linewidths=0.3, alpha=alpha)
        ax.add_collection3d(poly)



def plot_3d(run: dict):

    fig = plt.figure()
    fig.set_size_inches(10, 8)
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(True)
    ax.zaxis.pane.fill = False
    ax.zaxis._axinfo["grid"].update(color=(1, 1, 1, 0))

    # Draw gates
    for gate, quat in zip(run["gates"], run["gates_quat"]):
        draw_gate(ax, array(gate), array(quat))

    # Draw obstacles
    for obs in run["obstacles"]:
        draw_obstacle(ax, center=obs)

    # Plot trajectory spline
    spline = run["trajectory"](linspace(0, run["t_total"], 100))
    ax.plot(spline[:, 0], spline[:, 1], spline[:, 2], 'k--', linewidth=1.5, alpha=0.5, label="Spline")

    # Plot waypoints
    waypoints = array(run["waypoints"])
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], color="orange", s=30, marker='o')

    # Plot flown path
    flown = array(run["flown_path"])
    ax.plot(flown[:, 0], flown[:, 1], flown[:, 2], 'b-', linewidth=2.5, label="Flugweg")

    # Tick-Marker bei Änderungen (z. B. Gate-Updates)
    for gate_id, log in enumerate(run["gate_log"]):
        for tick, pos, _ in log:
            if tick < len(flown):
                p = flown[tick]
                ax.scatter(*p, color='magenta', s=20)
                ax.text(*p, f"T{tick}", fontsize=6, color='magenta')

    plt.title(f"Laufzeit: {run['time']:.2f}s", fontsize=14, weight='bold')
    ax.legend(loc='upper left', fontsize=10)
    plt.show(block=False)
    plt.pause(0.1)









