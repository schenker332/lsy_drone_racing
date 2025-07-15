import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameter aus dem MPC Controller
T_HORIZON = 1.5
N = 40
dt = T_HORIZON / N  # ≈ 0.0375
freq = 50  # Typische Frequenz

print(f"Parameter:")
print(f"dt = {dt:.6f}")
print(f"freq = {freq}")
print(f"dt * freq = {dt * freq:.6f}")

# Lade Daten aus der CSV-Datei
csv_path = "logs/run_20250712_121750/state_log.csv"
try:
    df = pd.read_csv(csv_path)
    print(f"\nCSV-Datei geladen: {csv_path}")
    print(f"Spalten in der CSV: {df.columns.tolist()}")
    
    # Berechne Gesamtgeschwindigkeit aus vx, vy, vz
    # speed = sqrt(vx² + vy² + vz²)
    df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
    
    # Verwende v_theta Werte
    v_theta_values = df['v_theta'].values
    time_values = df['time'].values
    speed_values = df['speed'].values
    
    print(f"\nErste 10 Werte:")
    for i in range(10):
        print(f"time={time_values[i]:.2f}, vx={df['vx'][i]:.3f}, vy={df['vy'][i]:.3f}, vz={df['vz'][i]:.3f}, speed={speed_values[i]:.3f}, v_theta={v_theta_values[i]:.6f}")

except FileNotFoundError:
    print(f"CSV-Datei nicht gefunden: {csv_path}")
    exit()

# Berechne t für jeden v_theta Wert
# t = 1 / (v_theta * dt * freq)
t_values = 1.0 / (v_theta_values * dt * freq)

# Plot erstellen
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: t_values über Zeit
ax1.plot(time_values, t_values, 'b-', linewidth=1, label='Berechnete t Werte')
ax1.set_xlabel('Zeit [s]')
ax1.set_ylabel('t Werte')
ax1.set_title('Berechnete t Werte über die Zeit')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Geschwindigkeit über Zeit
ax2.plot(time_values, speed_values, 'r-', linewidth=1, label='Gesamtgeschwindigkeit')
ax2.set_xlabel('Zeit [s]')
ax2.set_ylabel('Geschwindigkeit [m/s]')
ax2.set_title('Gesamtgeschwindigkeit über die Zeit')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: v_theta über Zeit
ax3.plot(time_values, v_theta_values, 'g-', linewidth=1, label='v_theta')
ax3.set_xlabel('Zeit [s]')
ax3.set_ylabel('v_theta')
ax3.set_title('v_theta über die Zeit')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Geschwindigkeitskomponenten
ax4.plot(time_values, df['vx'], 'r-', linewidth=1, alpha=0.7, label='vx')
ax4.plot(time_values, df['vy'], 'g-', linewidth=1, alpha=0.7, label='vy') 
ax4.plot(time_values, df['vz'], 'b-', linewidth=1, alpha=0.7, label='vz')
ax4.plot(time_values, speed_values, 'k-', linewidth=2, label='Gesamtgeschwindigkeit')
ax4.set_xlabel('Zeit [s]')
ax4.set_ylabel('Geschwindigkeit [m/s]')
ax4.set_title('Geschwindigkeitskomponenten')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.show()

print(f"\nZusammenfassung:")
print(f"Anzahl Datenpunkte: {len(v_theta_values)}")
print(f"Minimum t: {np.min(t_values):.2f}")
print(f"Maximum t: {np.max(t_values):.2f}")
print(f"Max Geschwindigkeit: {np.max(speed_values):.2f} m/s")
print(f"Durchschnitt Geschwindigkeit: {np.mean(speed_values):.2f} m/s")