import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.location import Location
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import os
import sys

# Get absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')

# Create directories if they don't exist
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_and_simulate_pvlib(tilt_angle=20, azimuth_angle=180, num_panels=10):
    """
    Load NSRDB data and simulate PV system using pvlib
    
    Parameters:
    - tilt_angle: Panel tilt angle in degrees
    - azimuth_angle: Panel azimuth angle in degrees (180 = south-facing)
    - num_panels: Number of panels to simulate
    
    Returns:
    - DataFrame with simulation results
    """
    
    # Load NSRDB data (skip first 2 metadata rows)
    nsrdb_path = os.path.join(DATA_RAW, 'nsrdb.csv')
    df = pd.read_csv(nsrdb_path, skiprows=2)
    
    # Build datetime index
    df["time"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
    df = df.set_index("time")
    
    # Rename and prepare columns for pvlib (ensure lowercase)
    df = df.rename(columns={
        "GHI": "ghi",
        "DNI": "dni", 
        "DHI": "dhi",
        "Temperature": "temp_air",
        "Wind Speed": "wind_speed",
        "Pressure": "pressure"
    })
    
    # Handle missing DHI column if not present
    if "dhi" not in df.columns and "DHI" in df.columns:
        df["dhi"] = df["DHI"]
    elif "dhi" not in df.columns:
        # Estimate DHI from GHI and DNI using simple model
        solar_zenith = 30  # rough estimate, could be calculated more precisely
        df["dhi"] = df["ghi"] - df["dni"] * np.cos(np.radians(solar_zenith))
        df["dhi"] = df["dhi"].clip(lower=0)  # ensure non-negative
    
    # Create location (Chennai coordinates)
    location = Location(latitude=13.05, longitude=80.25, tz="Asia/Kolkata")
    
    # Temperature model parameters
    temp_params = TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
    
    # Define PVSystem with configurable tilt and azimuth
    system = PVSystem(
        surface_tilt=tilt_angle,
        surface_azimuth=azimuth_angle,
        module_parameters={"pdc0": 250, "gamma_pdc": -0.004},  # 250W panel
        inverter_parameters={"pdc0": 250 * num_panels},
        temperature_model_parameters=temp_params,
        racking_model="open_rack",
        module_type="glass_glass"
    )
    
    # Create ModelChain for simulation
    mc = ModelChain(system, location, aoi_model="physical", spectral_model="no_loss")
    
    try:
        # Run the simulation
        mc.run_model(df)
        
        # Extract results
        results_df = pd.DataFrame(index=df.index)
        results_df["ac_power"] = mc.results.ac.fillna(0)  # AC power output in W
        results_df["dc_power"] = mc.results.dc.fillna(0)  # DC power output in W  
        results_df["energy_kwh"] = results_df["ac_power"] / 1000  # Convert to kWh (assuming hourly data)
        
        # Add weather data for reference
        results_df["ghi"] = df["ghi"]
        results_df["dni"] = df["dni"] 
        results_df["temp_air"] = df["temp_air"]
        
        # Scale by number of panels
        results_df["ac_power"] *= num_panels
        results_df["dc_power"] *= num_panels
        results_df["energy_kwh"] *= num_panels
        
        print(f"✅ PVLib simulation complete!")
        print(f"   System: {num_panels} panels, {tilt_angle}° tilt, {azimuth_angle}° azimuth")
        print(f"   Annual Energy: {results_df['energy_kwh'].sum():.2f} kWh")
        print(f"   Peak Power: {results_df['ac_power'].max():.2f} W")
        
        return results_df
        
    except Exception as e:
        print(f"❌ PVLib simulation failed: {e}")
        return None


if __name__ == "__main__":
    # Sweep through tilt angles 0° to 60° in steps of 5°
    results_summary = []
    best_angle, best_energy = None, 0
    
    for tilt in range(0, 61, 5):
        results = load_and_simulate_pvlib(tilt_angle=tilt, azimuth_angle=180, num_panels=10)
        if results is not None:
            annual_energy = results["energy_kwh"].sum()
            results_summary.append((tilt, annual_energy))
            if annual_energy > best_energy:
                best_energy = annual_energy
                best_angle = tilt
            # Save each tilt's detailed output if needed
            output_path = os.path.join(DATA_PROCESSED, f'pvlib_results_tilt{tilt}.csv')
            results.to_csv(output_path)
    
    # Save summary CSV
    summary_df = pd.DataFrame(results_summary, columns=["Tilt Angle (°)", "Annual Energy (kWh)"])
    summary_path = os.path.join(REPORTS_DIR, 'tilt_analysis.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Plot tilt vs energy
    plt.figure(figsize=(8, 6))
    plt.plot(summary_df["Tilt Angle (°)"], summary_df["Annual Energy (kWh)"], marker="o")
    plt.title("Annual Energy vs Tilt Angle")
    plt.xlabel("Tilt Angle (°)")
    plt.ylabel("Annual Energy (kWh)")
    plt.grid(True)
    plot_path = os.path.join(REPORTS_DIR, 'tilt_vs_energy.png')
    plt.savefig(plot_path, dpi=300)
    plt.show()
    
    print(f"\n🌞 Best tilt angle: {best_angle}° with {best_energy:.2f} kWh annual energy")
