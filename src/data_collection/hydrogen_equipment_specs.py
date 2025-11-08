"""
Hydrogen storage equipment specifications and safety parameters
Based on industry standards and manufacturer data
"""

import pandas as pd
import numpy as np

def create_hydrogen_equipment_database():
    """
    Create database of hydrogen equipment specifications
    """
    print("⚡ Creating hydrogen equipment specification database...")
    
    # Electrolyzer specifications (PEM type)
    electrolyzers = pd.DataFrame({
        'model': ['PEM-5kW', 'PEM-10kW', 'PEM-25kW', 'PEM-50kW'],
        'capacity_kw': [5, 10, 25, 50],
        'efficiency': [0.65, 0.70, 0.72, 0.75],
        'h2_production_rate_kg_hr': [0.10, 0.20, 0.50, 1.00],
        'water_consumption_l_hr': [0.9, 1.8, 4.5, 9.0],
        'operating_pressure_bar': [30, 30, 30, 35],
        'startup_time_min': [15, 10, 8, 5],
        'cost_inr': [400000, 750000, 1750000, 3500000],
        'lifetime_hours': [60000, 70000, 80000, 90000],
        'maintenance_interval_hours': [2000, 2500, 3000, 3500]
    })
    
    # Hydrogen storage tanks
    storage_tanks = pd.DataFrame({
        'model': ['Tank-5kg', 'Tank-10kg', 'Tank-25kg', 'Tank-50kg'],
        'capacity_kg': [5, 10, 25, 50],
        'pressure_bar': [700, 700, 700, 700],
        'volume_liters': [300, 600, 1500, 3000],
        'weight_kg': [80, 150, 350, 700],
        'cost_inr': [250000, 450000, 1000000, 1800000],
        'safety_cert': ['ISO 19881', 'ISO 19881', 'ISO 19881', 'ISO 19881'],
        'inspection_interval_days': [90, 90, 90, 90],
        'expected_lifetime_years': [20, 20, 20, 20]
    })
    
    # Fuel cells (PEM type)
    fuel_cells = pd.DataFrame({
        'model': ['FC-5kW', 'FC-10kW', 'FC-25kW', 'FC-50kW'],
        'rated_power_kw': [5, 10, 25, 50],
        'efficiency': [0.50, 0.52, 0.55, 0.58],
        'h2_consumption_kg_hr': [0.11, 0.22, 0.52, 1.00],
        'startup_time_sec': [30, 25, 20, 15],
        'operating_temp_c': [60, 65, 70, 75],
        'cost_inr': [500000, 950000, 2200000, 4200000],
        'lifetime_hours': [40000, 45000, 50000, 60000],
        'maintenance_interval_hours': [1500, 2000, 2500, 3000]
    })
    
    # Compressor specifications (for pressurizing H2)
    compressors = pd.DataFrame({
        'model': ['Comp-5kW', 'Comp-10kW', 'Comp-25kW'],
        'power_kw': [5, 10, 25],
        'flow_rate_kg_hr': [0.5, 1.0, 2.5],
        'compression_ratio': [700/30, 700/30, 700/30],  # From electrolyzer to tank
        'efficiency': [0.85, 0.87, 0.90],
        'cost_inr': [350000, 650000, 1500000],
        'noise_level_db': [65, 70, 75]
    })
    
    # Safety equipment
    safety_equipment = pd.DataFrame({
        'equipment': [
            'H2 Leak Detector',
            'Flame Detector',
            'Explosion-proof Ventilation',
            'Emergency Shutoff Valve',
            'Pressure Relief Valve',
            'Fire Suppression System'
        ],
        'cost_inr': [50000, 75000, 200000, 40000, 30000, 150000],
        'maintenance_interval_months': [6, 6, 12, 12, 12, 6],
        'mandatory': [True, True, True, True, True, False]
    })
    
    # Safety incidents database (for risk modeling)
    # Based on DOE Hydrogen incident database
    safety_incidents = pd.DataFrame({
        'incident_type': [
            'H2 Leak - Minor',
            'H2 Leak - Major',
            'Overpressure Event',
            'Equipment Failure',
            'Fire',
            'Explosion'
        ],
        'probability_per_year': [0.05, 0.01, 0.008, 0.02, 0.003, 0.001],
        'severity_score': [2, 7, 5, 4, 8, 10],  # 1-10 scale
        'consequence_cost_inr': [10000, 100000, 50000, 75000, 500000, 2000000],
        'mitigation_effectiveness': [0.90, 0.95, 0.98, 0.85, 0.92, 0.99]
    })
    
    return {
        'electrolyzers': electrolyzers,
        'storage_tanks': storage_tanks,
        'fuel_cells': fuel_cells,
        'compressors': compressors,
        'safety_equipment': safety_equipment,
        'safety_incidents': safety_incidents
    }

def calculate_safety_risk_score(equipment_config):
    """
    Calculate overall safety risk score for a given configuration
    """
    # Load safety incidents
    db = create_hydrogen_equipment_database()
    incidents = db['safety_incidents']
    
    # Calculate expected annual risk
    base_risk = (incidents['probability_per_year'] * 
                 incidents['severity_score']).sum()
    
    # Apply mitigation effectiveness
    mitigated_risk = base_risk * (1 - incidents['mitigation_effectiveness'].mean())
    
    # Adjust based on equipment quality and maintenance
    quality_factor = equipment_config.get('quality_factor', 0.8)  # 1.0 = perfect
    maintenance_factor = equipment_config.get('maintenance_factor', 0.9)
    
    final_risk = mitigated_risk / (quality_factor * maintenance_factor)
    
    return final_risk

def save_equipment_database(db_dict):
    """Save all equipment databases"""
    import os
    os.makedirs("../data/raw/", exist_ok=True)
    
    for name, df in db_dict.items():
        output_path = f"../data/raw/hydrogen_{name}.csv"
        df.to_csv(output_path, index=False)
        print(f"💾 Saved {name}: {output_path}")

if __name__ == "__main__":
    # Create equipment database
    db = create_hydrogen_equipment_database()
    
    # Display summaries
    print("\n⚡ ELECTROLYZER OPTIONS:")
    print(db['electrolyzers'][['model', 'capacity_kw', 'efficiency', 'cost_inr']])
    
    print("\n🔋 STORAGE TANK OPTIONS:")
    print(db['storage_tanks'][['model', 'capacity_kg', 'pressure_bar', 'cost_inr']])
    
    print("\n⚡ FUEL CELL OPTIONS:")
    print(db['fuel_cells'][['model', 'rated_power_kw', 'efficiency', 'cost_inr']])
    
    print("\n🛡️ SAFETY EQUIPMENT:")
    print(db['safety_equipment'][['equipment', 'cost_inr', 'mandatory']])
    
    print("\n⚠️ SAFETY INCIDENTS (Risk Analysis):")
    print(db['safety_incidents'][['incident_type', 'probability_per_year', 'severity_score']])
    
    # Calculate example risk score
    config = {'quality_factor': 0.9, 'maintenance_factor': 0.95}
    risk = calculate_safety_risk_score(config)
    print(f"\n📊 Example Safety Risk Score: {risk:.4f}")
    print(f"   Target threshold: < 0.01 (1%)")
    print(f"   Status: {'✅ SAFE' if risk < 0.01 else '⚠️ NEEDS IMPROVEMENT'}")
    
    # Save
    save_equipment_database(db)
