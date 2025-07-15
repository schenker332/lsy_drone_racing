import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

# ------------------------------------------------------------------
# Konfiguration: Nur hier Pfad zum Run-Ordner anpassen
# ------------------------------------------------------------------

run_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/run_20250712_140004"

# Pfade zu den CSV-Dateien
state_csv   = f"{run_dir}/state_log.csv"
gates_csv   = f"{run_dir}/final_gates.csv"

# ------------------------------------------------------------------
# 1) State-Log einlesen und prüfen
# ------------------------------------------------------------------
df = pd.read_csv(state_csv)

required_cols = {'time', 'vx', 'vy', 'vz', 'v_theta', 'ref_x', 'ref_y', 'ref_z'}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Fehlende Spalten in CSV: {missing}")

# ------------------------------------------------------------------
# 2) Grundgrößen berechnen
# ------------------------------------------------------------------
# Speed der Drohne
df['speed_drone'] = np.sqrt(df.vx**2 + df.vy**2 + df.vz**2)

# Arrays für Zeit, Theta, Referenzpunkte
time_vals    = df.time.values
v_theta_vals = df.v_theta.values
ref_pts      = df[['ref_x', 'ref_y', 'ref_z']].values

# MPC-Parameter (falls du sie ändern möchtest)
T_HORIZON = 1.5
N         = 40
dt        = T_HORIZON / N
freq      = 50

# ------------------------------------------------------------------
# 3) Länge des Referenzpfads berechnen
# ------------------------------------------------------------------
d_ref = np.diff(ref_pts, axis=0)
segment_lengths = np.linalg.norm(d_ref, axis=1)
path_length = segment_lengths.sum()
print(f"Gesamtlänge Referenzpfad: {path_length:.3f} m")

# ------------------------------------------------------------------
# 4) Ref-Speed Variante A (ideal)
# ------------------------------------------------------------------
# t = 1/(v_theta * dt * freq)
t_vals = 1.0 / (v_theta_vals * dt * freq)
speed_ref_A = path_length / t_vals

# ------------------------------------------------------------------
# 5) Ref-Speed Variante B (Δs/Δt)
# ------------------------------------------------------------------
dt_real     = np.diff(time_vals)
speed_ref_B = np.empty_like(time_vals)
speed_ref_B[0] = np.nan
speed_ref_B[1:] = segment_lengths / dt_real

# Durchschnitt der gemessenen Referenzgeschwindigkeit
avg_speed_ref_B = np.nanmean(speed_ref_B)
print(f"Durchschnittliche Ref-Speed B: {avg_speed_ref_B:.2f} m/s")

# ------------------------------------------------------------------
# 6) Gates einlesen und Zeitpunkte bestimmen
# ------------------------------------------------------------------
gates_df = pd.read_csv(gates_csv)
gate_times = []
for _, gate in gates_df.iterrows():
    gate_pos = np.array([gate.x, gate.y, gate.z])
    dists = np.linalg.norm(ref_pts - gate_pos, axis=1)
    idx_closest = np.argmin(dists)
    gate_times.append((int(gate.gate_idx), time_vals[idx_closest]))

# 7) Plotten aller Grafiken — jetzt nur noch 1 Reihe, 2 Spalten
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
ax1, ax2 = axs

# 7.1 Ideale Ref-Speed vs. Drohne
ax1.plot(time_vals, speed_ref_A, label='Ref-Speed (L/t)', linewidth=1.2)
ax1.plot(time_vals, df.speed_drone,       label='Drohne',     alpha=0.7)
ax1.set_title('Ideale Ref-Speed vs. Drohne')
ax1.set_xlabel('Zeit [s]')
ax1.set_ylabel('Geschwindigkeit [m/s]')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 7.2 Gemessene Ref-Speed vs. Drohne + Gates + Durchschnitt
ax2.plot(time_vals, speed_ref_B, label='Ref-Speed (Δs/Δt)', linewidth=1.2)
ax2.plot(time_vals, df.speed_drone,       label='Drohne',          alpha=0.7)
for gate_idx, gt in gate_times:
    ax2.axvline(gt, color='gray', linestyle='--', alpha=0.6)
    ax2.text(gt, ax2.get_ylim()[1]*0.9, f"G{gate_idx}",
             rotation=90, va='top', ha='center', alpha=0.8)
ax2.axhline(avg_speed_ref_B, color='blue', linestyle='-.', linewidth=1.5,
            label=f'Durchschnitt {avg_speed_ref_B:.2f} m/s')
ax2.text(0.99, avg_speed_ref_B, f"{avg_speed_ref_B:.2f} m/s",
         va='center', ha='right', backgroundcolor='white', alpha=0.8)

# Dynamische Y-Limits, kein overhang:
ymax = max(np.nanmax(speed_ref_B), np.nanmax(df.speed_drone)) * 1.05
ax2.set_ylim(0, ymax)

ax2.set_title('Gemessene Ref-Speed vs. Drohne (mit Gates & Durchschnitt)')
ax2.set_xlabel('Zeit [s]')
ax2.set_ylabel('Geschwindigkeit [m/s]')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()

# Speichern
out_png = os.path.join(run_dir, "speed_plot_top_only.png")
fig.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"➔ Plot gespeichert unter: {out_png}")
