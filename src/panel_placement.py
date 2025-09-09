# src/panel_placement.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load processed solar data
data_path = "../data/processed/solar_processed.csv"
df = pd.read_csv(data_path, parse_dates=["time"])

# Use ninja_pv (already normalized kW/kWp)
df = df.dropna(subset=["ninja_pv"])  # remove missing rows
df["solar_potential"] = df["ninja_pv"]

# Define a simple grid for potential panel locations
grid_size = (10, 10)  # 10x10 possible positions
panel_efficiency = 0.18  # 18%
panel_area = 1.6  # m^2 per panel

# Simulate placement by assigning random efficiency variations
np.random.seed(42)
placement_matrix = np.random.uniform(0.8, 1.0, size=grid_size)

# Calculate average solar potential over time
avg_potential = df["solar_potential"].mean()

# Energy output per panel position
energy_matrix = placement_matrix * avg_potential * panel_efficiency * panel_area

# Find best locations (top 10 positions)
flat_indices = np.argsort(energy_matrix.ravel())[::-1][:10]
best_positions = np.array(np.unravel_index(flat_indices, grid_size)).T

print("🔋 Top 10 panel placement positions (row, col):")
for pos in best_positions:
    print(tuple(pos))

# Visualization
plt.figure(figsize=(8,6))
plt.imshow(energy_matrix, cmap="viridis", origin="lower")
plt.colorbar(label="Relative Energy Output (kWh)")
plt.scatter(best_positions[:,1], best_positions[:,0], c="red", marker="x", label="Best")
plt.title("Optimal Panel Placement Grid")
plt.xlabel("Column")
plt.ylabel("Row")
plt.legend()
plt.savefig("../reports/panel_placement.png", dpi=300)
plt.show()
