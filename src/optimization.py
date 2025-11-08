import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')

# Parameters
installation_cost_per_kw = 60000
tariff_per_kwh = 8
annual_om_percent = 0.015

# Create reports directory if it doesn't exist
os.makedirs(REPORTS_DIR, exist_ok=True)

# Load energy estimation - use master dataset if energy_estimation.csv doesn't exist
energy_file = os.path.join(REPORTS_DIR, "energy_estimation.csv")
if not os.path.exists(energy_file):
    print("⚠️ energy_estimation.csv not found. Using master_dataset.csv...")
    master_path = os.path.join(DATA_PROCESSED, "master_dataset.csv")
    df_master = pd.read_csv(master_path)
    # Calculate total annual energy from master dataset
    # Use the correct column name present in master_dataset: 'energy_kwh'
    if 'energy_kwh' in df_master.columns:
        base_energy = df_master['energy_kwh'].sum()  # Total annual energy in kWh
    elif 'pv_energy_kw' in df_master.columns:
        # fallback for older datasets
        base_energy = df_master['pv_energy_kw'].sum()
    else:
        raise KeyError(f"Neither 'energy_kwh' nor 'pv_energy_kw' found in {master_path}")
    print(f"📊 Calculated base energy: {base_energy:.2f} kWh/year")
else:
    df = pd.read_csv(energy_file)
    base_energy = df['Total Annual Energy (kWh)'].iloc[0]

base_system_kw = 10   # corresponds to your earlier assumption

scenarios = []
for system_kw in range(5, 55, 5):  # from 5 kW to 50 kW
    # scale energy linearly
    annual_energy = base_energy * (system_kw / base_system_kw)
    capex = system_kw * installation_cost_per_kw
    annual_savings = annual_energy * tariff_per_kwh
    annual_om_cost = capex * annual_om_percent
    net_savings = annual_savings - annual_om_cost
    payback = capex / net_savings
    roi = (net_savings / capex) * 100

    scenarios.append([system_kw, annual_energy, capex, net_savings, payback, roi])

# Save results
df_out = pd.DataFrame(scenarios, columns=[
    "System Size (kW)", "Annual Energy (kWh)", "CAPEX (₹)",
    "Net Annual Savings (₹)", "Payback (years)", "ROI (%)"
])
results_path = os.path.join(REPORTS_DIR, "optimization_results.csv")
df_out.to_csv(results_path, index=False)

# Plot Energy vs System Size
plt.figure(figsize=(7,5))
plt.plot(df_out["System Size (kW)"], df_out["Annual Energy (kWh)"], marker="o")
plt.xlabel("System Size (kW)")
plt.ylabel("Annual Energy (kWh)")
plt.title("Energy Output vs System Size")
plt.grid(True)
energy_plot = os.path.join(REPORTS_DIR, "optimization_energy.png")
plt.savefig(energy_plot, dpi=300)

# Plot Payback vs System Size
plt.figure(figsize=(7,5))
plt.plot(df_out["System Size (kW)"], df_out["Payback (years)"], marker="o", color="orange")
plt.xlabel("System Size (kW)")
plt.ylabel("Payback (years)")
plt.title("Payback Period vs System Size")
plt.grid(True)
payback_plot = os.path.join(REPORTS_DIR, "optimization_payback.png")
plt.savefig(payback_plot, dpi=300)

print(f"📊 Optimization complete! Results saved in {results_path}")
