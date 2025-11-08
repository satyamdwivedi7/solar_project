#!/usr/bin/env python3
"""
Hybrid storage simulation:
 - Loads pvlib_results.csv (AC power)
 - Generates synthetic load
 - Simulates battery + hydrogen storage (electrolyzer + fuel cell)
 - Saves results to ../reports/hybrid_results.csv and plots
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Setup absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')

# Ensure directories exist
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---- Configurable system parameters ----
BATTERY_CAPACITY_KWH = 50.0        # kWh (installed battery energy capacity)
BATTERY_SOC_INIT = 0.2 * BATTERY_CAPACITY_KWH  # initial SOC (kWh)
BATTERY_ROUNDTRIP_EFF = 0.90       # roundtrip efficiency
# split roundtrip efficiency into charge/discharge factors
BATTERY_CHARGE_EFF = np.sqrt(BATTERY_ROUNDTRIP_EFF)
BATTERY_DISCHARGE_EFF = np.sqrt(BATTERY_ROUNDTRIP_EFF)
BATTERY_MAX_CHARGE_RATE_KW = 25.0  # kW (max charging power)
BATTERY_MAX_DISCHARGE_RATE_KW = 25.0

H2_TANK_CAPACITY_KWH = 500.0       # kWh-equivalent stored as H2 LHV energy
H2_SOC_INIT = 0.0                  # initial H2 stored (kWh equiv)
ELECTROLYZER_EFF = 0.65            # fraction (AC kWh -> kWh H2 energy)
FUELCELL_EFF = 0.52                # fraction (kWh H2 energy -> kWh electric)
H2_SAFETY_BUFFER = 0.05            # fraction of tank reserved (do not fill beyond 1 - buffer)

# Simulation settings
PV_INPUT_PATH = os.path.join(DATA_PROCESSED, "pvlib_results.csv")
OUTPUT_CSV = os.path.join(REPORTS_DIR, "hybrid_results.csv")
PLOT_PROFILE = os.path.join(REPORTS_DIR, "hybrid_profile.png")
PLOT_SUMMARY = os.path.join(REPORTS_DIR, "hybrid_storage_summary.png")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---- Utility functions ----
def read_pv_series(path):
    # Read pvlib results - try to detect the AC column.
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    # If index isn't datetime, try to parse a 'time' column
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time")
        else:
            # attempt to parse first column
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                raise RuntimeError("Could not interpret datetime index in pvlib_results.csv")

    # detect plausible AC column name
    col = None
    for candidate in ["ac", "AC", "p_ac", "power_ac", "power"]:
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        # fallback: take first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise RuntimeError("pvlib_results.csv contains no numeric columns.")
        col = numeric_cols[0]

    pv_series = df[col].astype(float)

    # pvlib typically outputs W, convert to kW if values look large
    if pv_series.max() > 2000:  # if more than 2 kW typical (common threshold)
        pv_kw = pv_series / 1000.0
    else:
        pv_kw = pv_series.copy()  # already kW

    # ensure hourly frequency (or at least preserve timestamps)
    return pv_kw

def generate_synthetic_load(index, base_kwh=10.0, peak_kwh=25.0):
    """
    Create synthetic hourly demand (kW) for each timestamp in index.
    base_kwh: minimum demand (kW)
    peak_kwh: peak demand (kW)
    """
    hours = index.hour.to_numpy()
    day_of_year = index.dayofyear.to_numpy()

    # daily pattern: small at dawn, peak in evening ~19:00
    # use combination of sin and gaussian centered at 19:00 for evening peak
    daily_base = base_kwh + (peak_kwh - base_kwh) * (np.sin((hours - 6) / 24 * 2 * np.pi) ** 2)
    # add evening bump (Gaussian around 19:00)
    evening_bump = (peak_kwh - base_kwh) * np.exp(-0.5 * ((hours - 19) / 2.5) ** 2)

    # seasonal multiplier (cooling/heating)
    seasonal = 1.0 + 0.08 * np.cos((day_of_year - 180) / 365.0 * 2 * np.pi)  # ±8%

    # random noise ±8%
    noise = np.random.normal(1.0, 0.08, size=len(index))

    demand = (daily_base + evening_bump) * seasonal * noise
    # ensure non-negative
    demand = np.maximum(demand, 0.1)
    return demand  # units: kW (power) per hour

# ---- Core simulation ----
def simulate_hybrid(pv_kw, load_kw,
                    batt_cap=BATTERY_CAPACITY_KWH,
                    batt_soc_init=BATTERY_SOC_INIT,
                    batt_charge_eff=BATTERY_CHARGE_EFF,
                    batt_discharge_eff=BATTERY_DISCHARGE_EFF,
                    batt_max_charge_kw=BATTERY_MAX_CHARGE_RATE_KW,
                    batt_max_discharge_kw=BATTERY_MAX_DISCHARGE_RATE_KW,
                    h2_tank_cap=H2_TANK_CAPACITY_KWH,
                    h2_soc_init=H2_SOC_INIT,
                    electrolyzer_eff=ELECTROLYZER_EFF,
                    fuelcell_eff=FUELCELL_EFF,
                    h2_safety_buffer=H2_SAFETY_BUFFER):
    """
    pv_kw, load_kw: pandas Series indexed by timestamps (kW)
    returns DataFrame with per-timestep simulation results
    """
    index = pv_kw.index
    n = len(index)

    # state variables
    batt_soc = batt_soc_init  # kWh
    h2_soc = h2_soc_init      # kWh equivalent (LHV basis)

    # arrays to record
    used_pv = np.zeros(n)              # kW delivered from PV to load
    charge_batt = np.zeros(n)          # kW used to charge battery
    discharge_batt = np.zeros(n)       # kW from battery to load
    batt_soc_hist = np.zeros(n)        # kWh
    h2_produced = np.zeros(n)         # kWh H2 energy produced (LHV)
    h2_stored_hist = np.zeros(n)      # kWh
    h2_used = np.zeros(n)             # kWh H2 energy consumed (LHV)
    electrolyzer_input_kw = np.zeros(n) # kW electrical input to electrolyzer
    fuelcell_output_kw = np.zeros(n)    # kW electrical output from fuel cell
    unmet_load = np.zeros(n)          # kW unmet after storages
    pv_to_batt = np.zeros(n)
    pv_to_h2 = np.zeros(n)
    batt_to_h2 = np.zeros(n)

    # usable h2 tank cap considering safety buffer
    usable_h2_cap = h2_tank_cap * (1.0 - h2_safety_buffer)

    for i, t in enumerate(index):
        pv = float(pv_kw.iloc[i])    # kW available from PV this hour
        demand = float(load_kw.iloc[i])  # kW demand this hour

        # STEP 1: PV directly to load
        pv_to_load = min(pv, demand)
        used_pv[i] = pv_to_load
        demand_remain = demand - pv_to_load
        pv_remain = pv - pv_to_load

        # STEP 2: If demand_remain > 0, discharge battery (if possible)
        if demand_remain > 0:
            # available discharge from battery in this hour:
            avail_discharge_kwh = min(batt_soc, batt_max_discharge_kw)  # kWh available to discharge limited by max rate and SOC
            # convert to electrical delivered considering discharge eff
            deliverable = avail_discharge_kwh * batt_discharge_eff
            # but demand is in kW for this hour; use min
            discharge_power = min(deliverable, demand_remain)
            # update SOC by energy withdrawn before efficiency losses
            if batt_discharge_eff > 0:
                energy_removed_from_batt = discharge_power / batt_discharge_eff
            else:
                energy_removed_from_batt = 0.0
            batt_soc -= energy_removed_from_batt
            discharge_batt[i] = discharge_power
            demand_remain = max(0.0, demand_remain - discharge_power)

        # STEP 3: If still demand_remain > 0, use H2 via fuel cell
        if demand_remain > 0:
            # fuelcell requires h2 energy = demand_remain / fuelcell_eff
            if fuelcell_eff <= 0:
                possible_from_h2 = 0.0
            else:
                h2_needed = demand_remain / fuelcell_eff  # kWh H2 energy
                # available h2 energy
                h2_available = h2_soc
                h2_used_kwh = min(h2_available, h2_needed)
                # electrical output from fuel cell
                electrical_from_h2 = h2_used_kwh * fuelcell_eff
                h2_soc -= h2_used_kwh
                h2_used[i] = h2_used_kwh
                fuelcell_output_kw[i] = electrical_from_h2
                demand_remain = max(0.0, demand_remain - electrical_from_h2)

        # if any demand_remain now -> unmet
        unmet_load[i] = demand_remain

        # STEP 4: With PV surplus (pv_remain), first charge battery (subject to rate/capacity)
        if pv_remain > 0:
            # maximum charge possible this hour (kW -> kWh over 1 hour)
            max_charge_allowed = min(batt_max_charge_kw, batt_cap - batt_soc)
            # PV energy we can allocate to battery this hour (kW)
            pv_charge_power = min(pv_remain, max_charge_allowed)
            # account for charge efficiency (energy stored)
            energy_stored = pv_charge_power * batt_charge_eff
            # update SOC (but ensure not exceeding capacity)
            actual_stored = min(energy_stored, batt_cap - batt_soc)
            # back-calc actual PV power used for that stored energy (accounting eff)
            if batt_charge_eff > 0:
                pv_consumed_for_batt = actual_stored / batt_charge_eff
            else:
                pv_consumed_for_batt = 0.0
            batt_soc += actual_stored
            pv_remain -= pv_consumed_for_batt
            charge_batt[i] = pv_consumed_for_batt
            pv_to_batt[i] = pv_consumed_for_batt

        # STEP 5: Any remaining PV surplus -> electrolyzer -> H2 (subject to usable tank cap)
        if pv_remain > 0:
            # electrical power allocated to electrolyzer (kW)
            # limiting by available usable H2 tank capacity:
            usable_space = usable_h2_cap - h2_soc
            if usable_space <= 0:
                # tank full (respecting safety buffer), cannot produce more H2
                alloc_to_elec = 0.0
            else:
                # PV_remain is kW for one hour -> kWh
                possible_h2_produced_if_all = pv_remain * electrolyzer_eff
                # cap production so it doesn't overfill tank
                produce_kwh = min(possible_h2_produced_if_all, usable_space)
                # back-calc required electrical input (kWh)
                if electrolyzer_eff > 0:
                    elec_input_needed = produce_kwh / electrolyzer_eff
                else:
                    elec_input_needed = 0.0
                # we cannot use more PV than pv_remain
                elec_input = min(pv_remain, elec_input_needed)
                produced = elec_input * electrolyzer_eff
                h2_soc += produced
                h2_produced[i] = produced
                electrolyzer_input_kw[i] = elec_input
                pv_to_h2[i] = elec_input
                pv_remain -= elec_input

        # record H2 storage level and battery SOC
        batt_soc_hist[i] = batt_soc
        h2_stored_hist[i] = h2_soc

    # build dataframe
    df_out = pd.DataFrame(index=index)
    df_out["pv_kw"] = pv_kw
    df_out["load_kw"] = load_kw
    df_out["used_pv_kw"] = used_pv
    df_out["batt_charge_kw"] = charge_batt
    df_out["batt_discharge_kw"] = discharge_batt
    df_out["batt_soc_kwh"] = batt_soc_hist
    df_out["h2_produced_kwh"] = h2_produced
    df_out["h2_stored_kwh"] = h2_stored_hist
    df_out["h2_used_kwh"] = h2_used
    df_out["electrolyzer_input_kw"] = electrolyzer_input_kw
    df_out["fuelcell_output_kw"] = fuelcell_output_kw
    df_out["pv_to_batt_kw"] = pv_to_batt
    df_out["pv_to_h2_kw"] = pv_to_h2
    df_out["unmet_load_kw"] = unmet_load
    df_out["served_kw"] = df_out["load_kw"] - df_out["unmet_load_kw"]

    return df_out

# ---- Main execution ----
def main():
    # Load PV series
    if not os.path.exists(PV_INPUT_PATH):
        raise FileNotFoundError(f"PV input not found at {PV_INPUT_PATH}")

    pv_kw = read_pv_series(PV_INPUT_PATH)  # series indexed by datetime

    # Build synthetic load aligned with PV timestamps
    load_kw_arr = generate_synthetic_load(pv_kw.index, base_kwh=10.0, peak_kwh=25.0)
    load_kw = pd.Series(load_kw_arr, index=pv_kw.index)

    # Run hybrid simulation
    df_sim = simulate_hybrid(pv_kw, load_kw,
                             batt_cap=BATTERY_CAPACITY_KWH,
                             batt_soc_init=BATTERY_SOC_INIT,
                             batt_charge_eff=BATTERY_CHARGE_EFF,
                             batt_discharge_eff=BATTERY_DISCHARGE_EFF,
                             batt_max_charge_kw=BATTERY_MAX_CHARGE_RATE_KW,
                             batt_max_discharge_kw=BATTERY_MAX_DISCHARGE_RATE_KW,
                             h2_tank_cap=H2_TANK_CAPACITY_KWH,
                             h2_soc_init=H2_SOC_INIT,
                             electrolyzer_eff=ELECTROLYZER_EFF,
                             fuelcell_eff=FUELCELL_EFF,
                             h2_safety_buffer=H2_SAFETY_BUFFER)

    # Save results
    df_sim.reset_index().rename(columns={"index": "time"}).to_csv(OUTPUT_CSV, index=False)
    print("✅ Hybrid simulation complete. Saved:", OUTPUT_CSV)

    # Summary metrics
    total_load = df_sim["load_kw"].sum()
    total_served = df_sim["served_kw"].sum()
    percent_served = 100.0 * (total_served / total_load) if total_load > 0 else 0.0
    total_h2_produced = df_sim["h2_produced_kwh"].sum()
    total_h2_used = df_sim["h2_used_kwh"].sum()
    print(f"Total load (kWh-hours): {total_load:.1f}")
    print(f"Total served (kWh-hours): {total_served:.1f} ({percent_served:.2f}%)")
    print(f"Total H2 produced (kWh-eq): {total_h2_produced:.1f}")
    print(f"Total H2 used (kWh-eq): {total_h2_used:.1f}")

    # Plot a short interval for visualization (first 7 days or available)
    days_to_plot = 7
    hours_to_plot = min(len(df_sim), days_to_plot*24)
    dfp = df_sim.iloc[:hours_to_plot]

    plt.figure(figsize=(14,6))
    plt.plot(dfp.index, dfp["load_kw"], label="Load (kW)", linewidth=1)
    plt.plot(dfp.index, dfp["pv_kw"], label="PV (kW)", linewidth=1)
    plt.plot(dfp.index, dfp["batt_soc_kwh"], label="Battery SOC (kWh)", linewidth=1)
    plt.plot(dfp.index, dfp["h2_stored_kwh"], label="H2 Stored (kWh-eq)", linewidth=1)
    plt.legend(loc="upper right")
    plt.xlabel("Time")
    plt.ylabel("kW / kWh")
    plt.title("Hybrid System Profile (first {} days)".format(days_to_plot))
    plt.tight_layout()
    plt.savefig(PLOT_PROFILE, dpi=300)
    plt.close()

    # Summary bar plot: Produced H2 vs Used H2 vs Battery capacity vs H2 capacity
    plt.figure(figsize=(8,5))
    bars = ["Battery cap (kWh)", "H2 usable cap (kWh-eq)", "H2 produced (kWh-eq)", "H2 used (kWh-eq)"]
    values = [BATTERY_CAPACITY_KWH, H2_TANK_CAPACITY_KWH*(1-H2_SAFETY_BUFFER), total_h2_produced, total_h2_used]
    plt.bar(bars, values)
    plt.xticks(rotation=25, ha="right")
    plt.title("Storage Summary")
    plt.tight_layout()
    plt.savefig(PLOT_SUMMARY, dpi=300)
    plt.close()

    print("Plots saved:", PLOT_PROFILE, "and", PLOT_SUMMARY)

if __name__ == "__main__":
    main()
