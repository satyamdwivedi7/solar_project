#!/usr/bin/env python3
"""
optimizer_ga.py

Genetic Algorithm to optimize:
  - battery_capacity_kwh
  - h2_tank_capacity_kwh

Objective: minimize LCOE (annualized cost / annual energy served)
Constraint: reliability >= RELIABILITY_TARGET (percent of load served)
If reliability < target, heavy penalty is applied.

Requires: hybrid_storage_simulation.py in the same folder and its functions:
  - read_pv_series(path)
  - generate_synthetic_load(index, base_kwh, peak_kwh)
  - simulate_hybrid(pv_kw, load_kw, batt_cap=..., h2_tank_cap=...)
(adapted names correspond to the implementation I provided earlier).

Usage:
    cd src
    python optimizer_ga.py

Outputs:
  - ../reports/optimization_ga_results.csv    (generation history + best)
  - ../reports/optimizer_convergence.png
  - printed best solution
"""

import os
import sys
import math
import random
import time
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

# Add src directory to path for imports
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import the simulation helpers from your hybrid script.
# Make sure hybrid_storage_simulation.py exists in src/ and defines:
#   read_pv_series, generate_synthetic_load, simulate_hybrid
from hybrid_storage_simulation import read_pv_series, generate_synthetic_load, simulate_hybrid

# ---- Configurable GA parameters ----
POP_SIZE = 24
GENERATIONS = 40
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.25
ELITISM = 2        # keep top N into next generation

# ---- Design variable bounds (search space) ----
BATTERY_MIN = 5.0      # kWh
BATTERY_MAX = 500.0    # kWh

H2_MIN = 50.0          # kWh-eq
H2_MAX = 3000.0        # kWh-eq

# ---- Reliability constraint (fraction) ----
RELIABILITY_TARGET = 0.95  # 95% of load must be served annually (tunable)

# ---- Cost & techno-economic params used by objective LCOE ----
# All currency units consistent (e.g., INR or USD)
PV_COST_PER_KW = 60000.0          # ₹/kW installed (if needed)
BATTERY_COST_PER_KWH = 25000.0    # ₹/kWh (installed)
H2_TANK_COST_PER_KWH = 2000.0     # ₹/kWh-eq
ELECTROLYZER_COST_PER_KW = 40000.0  # ₹/kW_electrolyzer
FUELCELL_COST_PER_KW = 40000.0      # ₹/kW_fuelcell
BOS_FRACTION = 0.15

LIFE_BATTERY = 10
LIFE_H2_TANK = 20
LIFE_ELECTROLYZER = 15
LIFE_FUELCELL = 15
DISCOUNT_RATE = 0.08
ANALYSIS_YEARS = 20

# OPEX fractions (annual fraction of CAPEX)
OPEX_BATT_FRAC = 0.02
OPEX_H2_TANK_FRAC = 0.01
OPEX_ELECTROLYZER_FRAC = 0.03
OPEX_FUELCELL_FRAC = 0.03

# Electrolyzer / fuel cell efficiencies assumed in simulation (should match hybrid script)
ELECTROLYZER_EFF = 0.65
FUELCELL_EFF = 0.52

# PV input file
PV_INPUT_PATH = os.path.join(DATA_PROCESSED, "pvlib_results.csv")

# Output paths
REPORT_CSV = os.path.join(REPORTS_DIR, "optimization_ga_results.csv")
PLOT_CONVERGENCE = os.path.join(REPORTS_DIR, "optimizer_convergence.png")

# Random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---- Utility functions ----
def crf(rate, n):
    if rate == 0:
        return 1.0/n
    return rate * (1+rate)**n / ((1+rate)**n - 1)

def annualize(capex, life):
    return capex * crf(DISCOUNT_RATE, life)

# Objective: compute LCOE for given pair (battery_kwh, h2_kwh)
def evaluate_candidate(individual, pv_kw_series, load_series):
    """individual: (battery_kwh, h2_kwh)"""
    batt_cap_kwh = float(individual[0])
    h2_cap_kwh = float(individual[1])

    # run hybrid simulation with these sizes
    # simulate_hybrid signature in hybrid_storage_simulation uses batt_cap and h2_tank_cap args
    df_sim = simulate_hybrid(
        pv_kw_series,
        load_series,
        batt_cap=batt_cap_kwh,
        batt_soc_init=0.0,
        batt_charge_eff=math.sqrt(0.9),
        batt_discharge_eff=math.sqrt(0.9),
        batt_max_charge_kw=max(1.0, batt_cap_kwh),   # allow 1C default
        batt_max_discharge_kw=max(1.0, batt_cap_kwh),
        h2_tank_cap=h2_cap_kwh,
        h2_soc_init=0.0,
        electrolyzer_eff=ELECTROLYZER_EFF,
        fuelcell_eff=FUELCELL_EFF,
        h2_safety_buffer=0.05
    )

    # compute annual values (scale if dataset not exactly one year)
    hours = len(df_sim)
    annual_hours = 8760.0
    scale = annual_hours / max(hours, 1)
    annual_served_kwh = df_sim["served_kw"].sum() * scale
    total_load_kwh = df_sim["load_kw"].sum() * scale
    reliability = (annual_served_kwh / max(total_load_kwh, 1e-9))

    # derive simple power ratings for electrolyzer/fuelcell from sim maxs
    electrolyzer_power_kw = df_sim["electrolyzer_input_kw"].max() if "electrolyzer_input_kw" in df_sim.columns else min( (pv_kw_series.max()), 1.0)
    fuelcell_power_kw = df_sim["fuelcell_output_kw"].max() if "fuelcell_output_kw" in df_sim.columns else 0.0

    # CAPEX estimates
    # PV capex is not optimized here; we only optimize storage. But we need PV capacity for BOS etc.
    pv_peak_kw = pv_kw_series.max()
    pv_capex = pv_peak_kw * PV_COST_PER_KW / 1000.0  # if PV_COST_PER_KW is per kW, convert accordingly
    # Note: user may want to include PV capex in optimization; currently it's a fixed base cost.

    capex_batt = batt_cap_kwh * BATTERY_COST_PER_KWH
    capex_h2 = h2_cap_kwh * H2_TANK_COST_PER_KWH
    capex_elec = electrolyzer_power_kw * ELECTROLYZER_COST_PER_KW
    capex_fc = fuelcell_power_kw * FUELCELL_COST_PER_KW
    capex_bos = pv_capex * BOS_FRACTION

    total_capex = pv_capex + capex_batt + capex_h2 + capex_elec + capex_fc + capex_bos

    # annualize capex
    annual_pv = annualize(pv_capex, ANALYSIS_YEARS)
    annual_batt = annualize(capex_batt, LIFE_BATTERY)
    annual_h2 = annualize(capex_h2, LIFE_H2_TANK)
    annual_elec = annualize(capex_elec, LIFE_ELECTROLYZER)
    annual_fc = annualize(capex_fc, LIFE_FUELCELL)
    annual_bos = annualize(capex_bos, ANALYSIS_YEARS)
    annualized_capex = annual_pv + annual_batt + annual_h2 + annual_elec + annual_fc + annual_bos

    # OPEX
    annual_opex = (capex_batt * OPEX_BATT_FRAC +
                   capex_h2 * OPEX_H2_TANK_FRAC +
                   capex_elec * OPEX_ELECTROLYZER_FRAC +
                   capex_fc * OPEX_FUELCELL_FRAC)

    # compute LCOE (penalize if energy served is tiny)
    if annual_served_kwh <= 0:
        lcoe = 1e9
    else:
        lcoe = (annualized_capex + annual_opex) / annual_served_kwh

    # Apply constraint penalty if reliability below target
    PENALTY_FACTOR = 1e6
    if reliability < RELIABILITY_TARGET:
        # penalty increases with shortfall
        shortfall = max(0.0, RELIABILITY_TARGET - reliability)
        lcoe = lcoe + PENALTY_FACTOR * shortfall

    # We'll also return some diagnostics
    info = {
        "batt_cap_kwh": batt_cap_kwh,
        "h2_cap_kwh": h2_cap_kwh,
        "annual_served_kwh": annual_served_kwh,
        "reliability": reliability,
        "lcoe": lcoe,
        "total_capex": total_capex,
        "annual_opex": annual_opex
    }
    return lcoe, info

# ---- GA operators ----
def create_individual():
    b = random.uniform(BATTERY_MIN, BATTERY_MAX)
    h = random.uniform(H2_MIN, H2_MAX)
    return [b, h]

def mutate(ind):
    # gaussian mutation with clip
    if random.random() < 0.5:
        ind[0] += random.gauss(0, (BATTERY_MAX - BATTERY_MIN) * 0.06)
    else:
        ind[1] += random.gauss(0, (H2_MAX - H2_MIN) * 0.06)
    ind[0] = max(BATTERY_MIN, min(BATTERY_MAX, ind[0]))
    ind[1] = max(H2_MIN, min(H2_MAX, ind[1]))
    return ind

def crossover(a, b):
    # blend crossover (alpha)
    alpha = random.random()
    child1 = [alpha*a[0] + (1-alpha)*b[0], alpha*a[1] + (1-alpha)*b[1]]
    child2 = [alpha*b[0] + (1-alpha)*a[0], alpha*b[1] + (1-alpha)*a[1]]
    return child1, child2

def tournament_selection(pop, scores, k=TOURNAMENT_SIZE):
    selected_idx = random.sample(range(len(pop)), k)
    best = min(selected_idx, key=lambda i: scores[i])
    return pop[best][:]  # return copy

# ---- Main GA loop ----
def run_ga():
    # load PV series and create load
    pv_kw = read_pv_series(PV_INPUT_PATH)
    load_arr = generate_synthetic_load(pv_kw.index, base_kwh=10.0, peak_kwh=25.0)
    load_series = pd.Series(load_arr, index=pv_kw.index)

    # population init
    population = [create_individual() for _ in range(POP_SIZE)]
    history = []

    # evaluate initial population
    scores = []
    infos = []
    print("Evaluating initial population...")
    for ind in population:
        score, info = evaluate_candidate(ind, pv_kw, load_series)
        scores.append(score)
        infos.append(info)

    best_overall = None
    best_score = float("inf")
    start_time = time.time()

    for gen in range(1, GENERATIONS + 1):
        new_pop = []

        # elitism: keep best ELITISM
        sorted_idx = sorted(range(len(population)), key=lambda i: scores[i])
        for i in range(min(ELITISM, len(population))):
            new_pop.append(population[sorted_idx[i]][:])

        # generate rest of new population
        while len(new_pop) < POP_SIZE:
            # selection
            parent1 = tournament_selection(population, scores)
            parent2 = tournament_selection(population, scores)
            # crossover
            if random.random() < CROSSOVER_RATE:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]
            # mutation
            if random.random() < MUTATION_RATE:
                child1 = mutate(child1)
            if random.random() < MUTATION_RATE:
                child2 = mutate(child2)
            new_pop.append(child1)
            if len(new_pop) < POP_SIZE:
                new_pop.append(child2)

        # evaluate new population
        population = new_pop
        scores = []
        infos = []
        print(f"Generation {gen}: evaluating {len(population)} candidates...")
        for ind in population:
            score, info = evaluate_candidate(ind, pv_kw, load_series)
            scores.append(score)
            infos.append(info)

        # store history
        gen_best_idx = int(np.argmin(scores))
        gen_best_score = scores[gen_best_idx]
        gen_best_info = infos[gen_best_idx]
        history.append({
            "generation": gen,
            "best_score": gen_best_score,
            "best_batt": gen_best_info["batt_cap_kwh"],
            "best_h2": gen_best_info["h2_cap_kwh"],
            "best_reliability": gen_best_info["reliability"],
            "best_annual_served_kwh": gen_best_info["annual_served_kwh"]
        })

        # track overall best
        if gen_best_score < best_score:
            best_score = gen_best_score
            best_overall = gen_best_info.copy()

        print(f" Gen {gen} best LCOE (penalized): {gen_best_score:.3f}; reliability: {gen_best_info['reliability']:.4f}")

    elapsed = time.time() - start_time
    print(f"\nGA finished in {elapsed:.1f}s. Best penalized LCOE: {best_score:.3f}")
    print("Best solution (unpenalized metrics):")
    print(best_overall)

    # save history to CSV
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(REPORT_CSV, index=False)
    print("Saved generation history to:", REPORT_CSV)

    # plot convergence (best_score per generation)
    plt.figure(figsize=(8,5))
    plt.plot(hist_df["generation"], hist_df["best_score"], marker='o')
    plt.xlabel("Generation")
    plt.ylabel("Best penalized LCOE")
    plt.title("GA Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_CONVERGENCE, dpi=300)
    print("Saved convergence plot to:", PLOT_CONVERGENCE)

    return best_overall, hist_df

if __name__ == "__main__":
    best, history = run_ga()
