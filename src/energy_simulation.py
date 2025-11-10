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

# Load processed solar dataset
DATA_PATH = os.path.join(DATA_PROCESSED, "solar_processed.csv")
OUTPUT_CSV = os.path.join(REPORTS_DIR, "energy_estimation.csv")
OUTPUT_PLOT = os.path.join(REPORTS_DIR, "energy_profile.png")

def simulate_energy(panel_efficiency=0.18, panel_area=1.6, num_panels=10):
    """
    Use PVLib simulation results for energy estimation.
    - panel_efficiency: PV efficiency (used for reference, pvlib already calculated with efficiency)
    - panel_area: Area per panel in m^2 (used for reference)
    - num_panels: Total number of panels to simulate
    """

    # Load solar data with pvlib results
    df = pd.read_csv(DATA_PATH)
    
    # Ensure time column is datetime
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    
    # Use energy_kwh from pvlib simulation (already calculated with proper physics models)
    if "energy_kwh" in df.columns:
        # Scale the energy based on number of panels (pvlib was run with 10 panels)
        base_panels = 10  # Number of panels used in pvlib simulation
        scaling_factor = num_panels / base_panels
        df["energy_kWh"] = df["energy_kwh"] * scaling_factor
        print(f"📊 Using PVLib energy calculations (scaled for {num_panels} panels)")
    else:
        # Fallback: use GHI for manual calculation
        if "GHI" in df.columns and df["GHI"].notna().any():
            irradiance = df["GHI"].fillna(0)
        else:
            print("❌ No irradiance data available")
            return

        # Power output (W) = Irradiance (W/m²) × Area × Efficiency × Num_panels
        df["power_W"] = irradiance * panel_area * panel_efficiency * num_panels
        df["energy_kWh"] = df["power_W"] / 1000
        print(f"📊 Using manual GHI calculations")

    # Summaries
    total_energy = df["energy_kWh"].sum()
    avg_daily_energy = df.groupby(df["time"].dt.date)["energy_kWh"].sum().mean()

    # Save results
    results = pd.DataFrame({
        "Total Annual Energy (kWh)": [total_energy],
        "Average Daily Energy (kWh)": [avg_daily_energy],
        "Number of Panels": [num_panels],
        "Panel Efficiency": [panel_efficiency],
        "Panel Area (m2)": [panel_area]
    })
    results.to_csv(OUTPUT_CSV, index=False)

    # Plot energy profile
    plt.figure(figsize=(10,5))
    df.set_index("time")["energy_kWh"].resample("D").sum().plot()
    plt.title("Daily Energy Generation Profile")
    plt.ylabel("Energy (kWh)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.close()

    print(f"✅ Energy simulation complete!")
    print(f"   Total Energy: {total_energy:.2f} kWh/year")
    print(f"   Average Daily: {avg_daily_energy:.2f} kWh/day")
    print(f"   Results saved to {OUTPUT_CSV}")
    print(f"   Plot saved to {OUTPUT_PLOT}")


if __name__ == "__main__":
    simulate_energy(panel_efficiency=0.18, panel_area=1.6, num_panels=20)
