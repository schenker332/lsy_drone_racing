import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# ------------------------------------------------------------------
# Load CSV with logged reference points
# ------------------------------------------------------------------
csv_path = "logs/run_20250712_123750/state_log.csv"  # adjust if needed
df = pd.read_csv(csv_path)

# Ensure required columns exist
for col in ["ref_x", "ref_y", "ref_z"]:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in CSV. Available columns: {df.columns.tolist()}")

# ------------------------------------------------------------------
# Sample every 6th point for clarity
# ------------------------------------------------------------------
df_sample = df.iloc[::4]

# ------------------------------------------------------------------
# Create 3D plot of ref points
# ------------------------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
ax.plot(df_sample["ref_x"], df_sample["ref_y"], df_sample["ref_z"], marker="o", linestyle="-", label="Alle 6. Punkte")
ax.set_title("3D-Verteilung der Referenzpunkte")
ax.set_xlabel("ref_x [m]")
ax.set_ylabel("ref_y [m]")
ax.set_zlabel("ref_z [m]")
ax.legend()
ax.grid(True)

plt.show()
