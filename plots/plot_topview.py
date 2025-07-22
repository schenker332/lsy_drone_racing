#!/usr/bin/env python3
"""
plot_topview.py

Plottet für einen Run-Ordner in plots/data/<run_name>:
  - 2D‑Top‑Down‑Ansicht der Flugtrajektorie (px,py), eingefärbt nach Speed
  - Gates (rote Dreiecke) und Obstacles (schwarze Kreise)

Aufruf:
    python plots/plot_topview.py <run_name>
z.B.:
    python plots/plot_topview.py 16_31_DO_v4_vtheta_real_flight_with_obstacles
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fire

def read_states(state_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(state_csv)
    required = {"px", "py", "pz", "vx", "vy", "vz", "time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {state_csv}: {missing}")
    return df

def read_gates_and_obstacles(gates_obst_csv: Path):
    df = pd.read_csv(gates_obst_csv)
    last = df.iloc[-1]
    # Gate-Spalten erkennen: g0_x, g1_x, ...  
    gate_x = sorted(c for c in df.columns if c.startswith("g") and c.endswith("_x"))
    obst_x = sorted(c for c in df.columns if c.startswith("o") and c.endswith("_x"))
    gates = np.stack([[last[gx], last[gx[:-2] + "_y"], last[gx[:-2] + "_z"]] for gx in gate_x])
    obst  = np.stack([[last[ox], last[ox[:-2] + "_y"], last[ox[:-2] + "_z"]] for ox in obst_x])
    return gates, obst

def plot_topview(run_name: str):
    # Basis-Pfade
    base = Path(__file__).parent / "data"
    run_dir = base / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")

    # Erwartete Dateien
    state_csv      = run_dir / "run_xcurrent.csv"
    gates_obst_csv = run_dir / "run_gates_obst.csv"
    if not state_csv.exists() or not gates_obst_csv.exists():
        raise FileNotFoundError(f"Missing CSVs in {run_dir}")

    # Daten laden
    df = read_states(state_csv)
    gates, obst = read_gates_and_obstacles(gates_obst_csv)

    # Speed berechnen
    speed = np.sqrt(df.vx**2 + df.vy**2 + df.vz**2)

    # Plot
    fig, ax = plt.subplots(figsize=(8,8))
    sc = ax.scatter(df.px, df.py, c=speed, cmap="viridis", s=4, lw=0, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Speed [m/s]")

    # Gates
    ax.scatter(gates[:,0], gates[:,1], marker="^", c="red",
               s=100, edgecolor="k", label="Gates")
    # Obstacles
    ax.scatter(obst[:,0], obst[:,1], marker="o", c="black",
               s=50, label="Obstacles")

    ax.set_aspect("equal", "box")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Top-Down Flight: {run_name}")
    ax.legend(loc="upper right")

    out_png = run_dir / f"{run_name}_topview.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"➔ Saved plot to {out_png}")

if __name__ == "__main__":
    fire.Fire(plot_topview)
