# plot_speed.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_speed(run_dir: Path | str) -> None:
    """
    Ein einziger Plot mit:
      • drone speed
      • Ref‑Speed (L/t)
      • Ref‑Speed (Δs/Δt)
      • curvature
    Speichert speed_plot_combined.png in run_dir.
    """
    run_dir   = Path(run_dir)
    state_csv = run_dir / "state_log.csv"
    gates_csv = run_dir / "final_gates.csv"

    # CSV einlesen und prüfen
    df = pd.read_csv(state_csv)
    needed = {'time','vx','vy','vz','v_theta','ref_x','ref_y','ref_z','curvature'}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"Fehlende Spalten: {miss}")

    # Basisgrößen
    df['speed_drone'] = np.sqrt(df.vx**2 + df.vy**2 + df.vz**2)
    t = df.time.values
    v_theta = df.v_theta.values
    ref_pts = df[['ref_x','ref_y','ref_z']].values
    curvature = df.curvature.values

    # MPC‑Parameter
    T_H, N, freq = 1.5, 40, 50
    dt = T_H / N

    # Pfadlänge
    d_ref = np.diff(ref_pts, axis=0)
    seg_len = np.linalg.norm(d_ref, axis=1)
    L = seg_len.sum()

    # Ref‑Speed A (L/t)
    t_vals = 1/(v_theta * dt * freq)
    speed_ref_A = L / t_vals

    # Ref‑Speed B (Δs/Δt)
    dt_real = np.diff(t)
    speed_ref_B = np.empty_like(t)
    speed_ref_B[0] = np.nan
    speed_ref_B[1:] = seg_len / dt_real

    # Gemeinsamer Plot
    fig, ax1 = plt.subplots(figsize=(10,6))

    # linke Y‑Achse: Geschwindigkeiten
    ax1.plot(t, df.speed_drone,   label='Drone Speed', color='black', linewidth=1.5)
    ax1.plot(t, speed_ref_A,      label='Ref‑Speed (Course Speed for the complete course) / V_theta', linestyle='--')
    ax1.plot(t, speed_ref_B,      label='Ref‑Speed (Δs/Δt)', linestyle='-.')
    ax1.set_xlabel('Zeit [s]')
    ax1.set_ylabel('Geschwindigkeit [m/s]')
    ax1.grid(alpha=0.3)

    # rechte Y‑Achse: curvature
    ax2 = ax1.twinx()
    ax2.plot(t, curvature, color='magenta', linestyle=':', label='curvature')
    ax2.set_ylabel('Krümmung')

    # Kombinierte Legende
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax1.set_title('Drone Speed, Ref‑Speed A (L/t), Ref‑Speed B (Δs/Δt) und Curvature')

    plt.tight_layout()
    out = run_dir / "speed_plot_combined.png"
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"➔ Combined-Speed‑Plot abgelegt: {out}")


if __name__ == "__main__":
    import sys
    rd = Path(sys.argv[1]) if len(sys.argv)>1 else Path("logs/run_0000")
    plot_speed(rd)
