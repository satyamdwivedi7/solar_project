import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters (can be tuned)
ELECTROLYZER_EFF = 0.7       # 70% efficiency
FUELCELL_EFF = 0.5           # 50% efficiency
H2_TANK_CAPACITY = 5000      # kWh equivalent (storage)
BASE_LOAD = 10               # kWh minimum demand
PEAK_LOAD = 25               # kWh maximum demand
SEED = 42

np.random.seed(SEED)

def generate_synthetic_load(df):
    """Generate synthetic hourly demand profile (kWh)."""
    hours = df.index.hour
    days = df.index.dayofyear

    # Daily cycle: low at night, peak in evening
    daily_cycle = BASE_LOAD + (PEAK_LOAD - BASE_LOAD) * (
        np.sin((hours - 12) / 24 * 2 * np.pi) ** 2
    )

    # Seasonal effect (10% more demand in summer/winter extremes)
    seasonal = 1 + 0.1 * np.cos((days - 180) / 365 * 2 * np.pi)

    # Random variation ±10%
    noise = np.random.normal(1, 0.1, len(df))

    demand = daily_cycle * seasonal * noise
    return demand

def simulate_hydrogen(df):
    """Simulate hydrogen storage with PV + load balance."""
    h2_storage = 0
    storage_history = []
    net_load_history = []

    # Use energy_kwh from PVLib simulation instead of ninja_pv
    pv_column = "energy_kwh" if "energy_kwh" in df.columns else "ac_power"
    if pv_column == "ac_power":
        # Convert AC power (W) to energy (kWh) for hourly data
        pv_generation = df[pv_column] / 1000  # W to kWh
    else:
        pv_generation = df[pv_column]

    for pv_gen, demand in zip(pv_generation, df["load"]):
        surplus = pv_gen - demand

        if surplus > 0:
            # Convert surplus to hydrogen
            h2_produced = surplus * ELECTROLYZER_EFF
            h2_storage = min(H2_TANK_CAPACITY, h2_storage + h2_produced)
            net_load = demand  # fully met by PV
        else:
            # Need extra energy: draw from H2 storage
            h2_needed = abs(surplus) / FUELCELL_EFF
            if h2_storage >= h2_needed:
                h2_storage -= h2_needed
                net_load = demand  # fully met
            else:
                # Not enough hydrogen → unmet demand
                net_load = demand - (pv_gen + h2_storage * FUELCELL_EFF)
                h2_storage = 0

        storage_history.append(h2_storage)
        net_load_history.append(net_load)

    df["H2_Storage"] = storage_history
    df["Net_Load_Served"] = net_load_history
    df["PV_Generation"] = pv_generation  # Add for plotting
    return df

def main():
    # Load processed solar dataset
    df = pd.read_csv("../data/processed/solar_processed.csv", parse_dates=["time"], index_col="time")

    # Generate synthetic load
    df["load"] = generate_synthetic_load(df)

    # Determine which PV column to use
    pv_column = "energy_kwh" if "energy_kwh" in df.columns else "ac_power"

    # Simulate hydrogen storage system
    df = simulate_hydrogen(df)

    # Save results
    df.to_csv("../reports/hydrogen_results.csv")

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(df.index[:168], df["load"][:168], label="Load (kWh)", color="orange")
    plt.plot(df.index[:168], df["PV_Generation"][:168], label="PV Generation (kWh)", color="green")
    plt.plot(df.index[:168], df["H2_Storage"][:168], label="H2 Tank (kWh equiv.)", color="blue")
    plt.title("Hydrogen Storage Simulation (First Week) - PVLib Data")
    plt.xlabel("Time")
    plt.ylabel("Energy (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../reports/hydrogen_profile.png", dpi=300)
    plt.close()

    print("✅ Hydrogen simulation complete!")
    print(f"   PV Energy Source: {pv_column}")
    print(f"   Annual PV Generation: {df['PV_Generation'].sum():.2f} kWh")
    print(f"   Annual Load: {df['load'].sum():.2f} kWh")
    print(f"   Max H2 Storage Used: {df['H2_Storage'].max():.2f} kWh")
    print("   Results saved to ../reports/hydrogen_results.csv")
    print("   Plot saved to ../reports/hydrogen_profile.png")

if __name__ == "__main__":
    main()
