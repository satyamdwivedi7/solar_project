# src/panel_placement.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Setup absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')

# Ensure directories exist
os.makedirs(REPORTS_DIR, exist_ok=True)

# Load processed solar data with PVLib results
data_path = os.path.join(DATA_PROCESSED, "solar_processed.csv")
df = pd.read_csv(data_path, parse_dates=["time"])

# Use pvlib energy_kwh data (already calculated with proper physics)
if "energy_kwh" in df.columns:
    df = df.dropna(subset=["energy_kwh"])  # remove missing rows
    df["solar_potential"] = df["energy_kwh"]
    print("📊 Using PVLib energy calculations for panel placement")
elif "ac_power" in df.columns:
    df = df.dropna(subset=["ac_power"])
    df["solar_potential"] = df["ac_power"] / 1000  # Convert W to kW
    print("📊 Using PVLib AC power for panel placement")
else:
    # Fallback to GHI if available
    if "GHI" in df.columns:
        df = df.dropna(subset=["GHI"])
        df["solar_potential"] = df["GHI"] / 1000  # Normalize
        print("📊 Using GHI data for panel placement")
    else:
        print("❌ No suitable solar data found")
        exit(1)

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
