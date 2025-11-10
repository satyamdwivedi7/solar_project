"""
Optimal Tilt Angle Finder for Solar PV Systems
================================================
This script determines the optimal tilt angle for solar panels at a given location
by simulating PV energy production across different tilt angles.

Main Output: Optimal tilt angle in degrees for maximum annual energy production
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pvlib
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
LOCATION = {
    'name': 'Chennai, Tamil Nadu, India',
    'latitude': 13.05,
    'longitude': 80.25,
    'altitude': 16  # meters (coastal elevation)
}

# System parameters
PANEL_PARAMS = {
    'Pmp': 400,  # Watts at maximum power point
    'Vmp': 40,   # Voltage at maximum power
    'Imp': 10,   # Current at maximum power
    'Voc': 48,   # Open circuit voltage
    'Isc': 10.5, # Short circuit current
    'alpha_sc': 0.0005,  # Temperature coefficient for Isc
    'beta_voc': -0.003,  # Temperature coefficient for Voc
    'gamma_pmp': -0.004, # Temperature coefficient for Pmp
    'cells_in_series': 72
}

SYSTEM_PARAMS = {
    'num_panels': 20,
    'inverter_efficiency': 0.96,
    'dc_losses': 0.04,  # Soiling, wiring, etc.
    'azimuth': 180  # South-facing (180° in Northern Hemisphere)
}

# Tilt angles to test (degrees)
TILT_ANGLES = np.arange(0, 61, 5)  # 0° to 60° in 5° steps

def load_weather_data():
    """Load weather data from master dataset"""
    print("📊 Loading weather data...")
    
    # Try different possible paths for master_dataset
    possible_paths = [
        'data/processed/master_dataset.csv',
        '../data/processed/master_dataset.csv',
        os.path.join(os.path.dirname(__file__), '../data/processed/master_dataset.csv')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            
            # Rename columns to match PVLib expectations
            df_weather = df[['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed']].copy()
            df_weather.columns = ['GHI', 'DNI', 'DHI', 'Temperature', 'Wind Speed']
            
            print(f"✅ Loaded weather data from: {path}")
            print(f"   Records: {len(df_weather):,} hours")
            return df_weather
    
    raise FileNotFoundError(f"Could not find master_dataset.csv in any of these locations: {possible_paths}")

def simulate_pv_for_tilt(weather_data, tilt_angle):
    """
    Simulate PV energy production for a specific tilt angle
    
    Parameters:
    -----------
    weather_data : DataFrame
        Weather data with GHI, DNI, DHI, Temperature, Wind Speed
    tilt_angle : float
        Panel tilt angle in degrees
    
    Returns:
    --------
    float : Total annual energy production in kWh
    """
    
    # Create location object
    location = pvlib.location.Location(
        latitude=LOCATION['latitude'],
        longitude=LOCATION['longitude'],
        altitude=LOCATION['altitude'],
        name=LOCATION['name']
    )
    
    # Create datetime index
    times = pd.date_range(
        start='2023-01-01 00:00:00',
        end='2023-12-31 23:00:00',
        freq='h',
        tz='Asia/Kolkata'
    )
    
    # Ensure weather data has the right length
    if len(weather_data) > len(times):
        weather_data = weather_data.iloc[:len(times)]
    elif len(weather_data) < len(times):
        raise ValueError(f"Weather data has only {len(weather_data)} records, need {len(times)}")
    
    weather_data.index = times
    
    # Calculate solar position
    solar_position = location.get_solarposition(times)
    
    # Calculate POA (Plane of Array) irradiance using isotropic model (simpler, no dni_extra needed)
    poa_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_angle,
        surface_azimuth=SYSTEM_PARAMS['azimuth'],
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'],
        dni=weather_data['DNI'],
        ghi=weather_data['GHI'],
        dhi=weather_data['DHI'],
        model='isotropic'  # Simpler model that doesn't require dni_extra
    )
    
    # Cell temperature model
    temp_cell = pvlib.temperature.faiman(
        poa_global=poa_irradiance['poa_global'],
        temp_air=weather_data['Temperature'],
        wind_speed=weather_data['Wind Speed']
    )
    
    # Calculate DC power for single module
    photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = (
        pvlib.pvsystem.calcparams_desoto(
            poa_irradiance['poa_global'],
            temp_cell,
            alpha_sc=PANEL_PARAMS['alpha_sc'],
            a_ref=1.5,
            I_L_ref=PANEL_PARAMS['Isc'],
            I_o_ref=1e-10,
            R_sh_ref=1000,
            R_s=0.3,
            EgRef=1.121,
            dEgdT=-0.0002677
        )
    )
    
    # Calculate IV curve parameters
    dc_power = pvlib.pvsystem.max_power_point(
        photocurrent,
        saturation_current,
        resistance_series,
        resistance_shunt,
        nNsVth,
        method='newton'
    )
    
    # Scale to system size
    dc_power_system = dc_power['p_mp'] * SYSTEM_PARAMS['num_panels']
    
    # Apply DC losses
    dc_power_system *= (1 - SYSTEM_PARAMS['dc_losses'])
    
    # Convert to AC (inverter)
    ac_power = dc_power_system * SYSTEM_PARAMS['inverter_efficiency']
    
    # Clip negative values
    ac_power = ac_power.clip(lower=0)
    
    # Calculate total annual energy (kWh)
    annual_energy = ac_power.sum() / 1000  # Convert W to kW
    
    return annual_energy

def find_optimal_tilt():
    """
    Main function to find optimal tilt angle
    
    Returns:
    --------
    dict : Results containing optimal tilt, energy values, and statistics
    """
    
    print("\n" + "="*70)
    print("🌞 OPTIMAL TILT ANGLE FINDER FOR SOLAR PV SYSTEMS")
    print("="*70)
    print(f"\n📍 Location: {LOCATION['name']}")
    print(f"   Latitude: {LOCATION['latitude']}°N")
    print(f"   Longitude: {LOCATION['longitude']}°E")
    print(f"   Altitude: {LOCATION['altitude']} m")
    print(f"\n⚡ System Configuration:")
    print(f"   Panels: {SYSTEM_PARAMS['num_panels']} × {PANEL_PARAMS['Pmp']}W")
    print(f"   Total Capacity: {SYSTEM_PARAMS['num_panels'] * PANEL_PARAMS['Pmp'] / 1000:.2f} kW")
    print(f"   Azimuth: {SYSTEM_PARAMS['azimuth']}° (South-facing)")
    print(f"\n🔍 Testing {len(TILT_ANGLES)} tilt angles: {TILT_ANGLES[0]}° to {TILT_ANGLES[-1]}°")
    print("="*70 + "\n")
    
    # Load weather data
    weather_data = load_weather_data()
    
    # Simulate for each tilt angle
    results = []
    
    for i, tilt in enumerate(TILT_ANGLES):
        print(f"[{i+1}/{len(TILT_ANGLES)}] Simulating tilt angle: {tilt:2.0f}° ... ", end='', flush=True)
        
        try:
            annual_energy = simulate_pv_for_tilt(weather_data, tilt)
            results.append({
                'tilt_angle': tilt,
                'annual_energy_kwh': annual_energy
            })
            print(f"✅ {annual_energy:,.1f} kWh/year")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            continue
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Find optimal tilt
    optimal_idx = df_results['annual_energy_kwh'].idxmax()
    optimal_tilt = df_results.loc[optimal_idx, 'tilt_angle']
    optimal_energy = df_results.loc[optimal_idx, 'annual_energy_kwh']
    
    # Calculate statistics
    min_energy = df_results['annual_energy_kwh'].min()
    max_energy = df_results['annual_energy_kwh'].max()
    energy_gain = ((max_energy - min_energy) / min_energy) * 100
    
    # Compare to latitude rule of thumb
    latitude_tilt = LOCATION['latitude']
    latitude_energy = df_results[df_results['tilt_angle'].between(latitude_tilt-2.5, latitude_tilt+2.5)]['annual_energy_kwh'].max()
    improvement_vs_latitude = ((optimal_energy - latitude_energy) / latitude_energy) * 100
    
    # Print results
    print("\n" + "="*70)
    print("📊 OPTIMIZATION RESULTS")
    print("="*70)
    print(f"\n🎯 OPTIMAL TILT ANGLE: {optimal_tilt:.0f}°")
    print(f"   Annual Energy Production: {optimal_energy:,.1f} kWh/year")
    print(f"   Daily Average: {optimal_energy/365:.1f} kWh/day")
    print(f"\n📈 Performance Range:")
    print(f"   Minimum ({df_results.loc[df_results['annual_energy_kwh'].idxmin(), 'tilt_angle']:.0f}°): {min_energy:,.1f} kWh/year")
    print(f"   Maximum ({optimal_tilt:.0f}°): {max_energy:,.1f} kWh/year")
    print(f"   Gain from Optimization: {energy_gain:.1f}%")
    print(f"\n💡 Comparison:")
    print(f"   Latitude Rule ({latitude_tilt:.1f}°): {latitude_energy:,.1f} kWh/year")
    print(f"   Improvement vs Latitude Rule: {improvement_vs_latitude:+.2f}%")
    print("="*70 + "\n")
    
    # Save results
    output_dir = 'reports'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'optimal_tilt_analysis.csv')
    df_results.to_csv(output_file, index=False)
    print(f"💾 Results saved to: {output_file}")
    
    # Create visualizations
    create_visualizations(df_results, optimal_tilt, optimal_energy)
    
    return {
        'optimal_tilt': optimal_tilt,
        'optimal_energy': optimal_energy,
        'results_df': df_results,
        'latitude_comparison': improvement_vs_latitude
    }

def create_visualizations(df_results, optimal_tilt, optimal_energy):
    """Create visualization plots"""
    
    print("\n📊 Creating visualizations...")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Energy vs Tilt Angle
    ax1.plot(df_results['tilt_angle'], df_results['annual_energy_kwh'], 
             'b-o', linewidth=2, markersize=6, label='Energy Production')
    ax1.axvline(optimal_tilt, color='r', linestyle='--', linewidth=2, 
                label=f'Optimal: {optimal_tilt:.0f}°')
    ax1.axhline(optimal_energy, color='g', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Tilt Angle (degrees)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Annual Energy Production (kWh)', fontsize=12, fontweight='bold')
    ax1.set_title('Annual Energy vs Tilt Angle', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Annotate optimal point
    ax1.annotate(f'{optimal_energy:,.0f} kWh/yr',
                xy=(optimal_tilt, optimal_energy),
                xytext=(optimal_tilt+10, optimal_energy-200),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # Plot 2: Relative Performance (%)
    baseline_energy = df_results['annual_energy_kwh'].iloc[0]  # 0° tilt
    relative_performance = ((df_results['annual_energy_kwh'] - baseline_energy) / baseline_energy) * 100
    
    ax2.bar(df_results['tilt_angle'], relative_performance, 
            color=['green' if x == optimal_tilt else 'skyblue' for x in df_results['tilt_angle']],
            edgecolor='black', linewidth=1)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Tilt Angle (degrees)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Performance vs Horizontal (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Relative Performance Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Annotate optimal bar
    optimal_performance = relative_performance[df_results['tilt_angle'] == optimal_tilt].values[0]
    ax2.text(optimal_tilt, optimal_performance + 1, 
             f'+{optimal_performance:.1f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold', color='darkgreen')
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join('reports', 'optimal_tilt_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {output_file}")
    
    plt.show()

def main():
    """Main execution function"""
    try:
        results = find_optimal_tilt()
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\n🎯 Main Result: Optimal Tilt Angle = {results['optimal_tilt']:.0f}°")
        print(f"   Expected Annual Energy: {results['optimal_energy']:,.1f} kWh/year")
        print(f"\n📁 Output Files:")
        print(f"   - reports/optimal_tilt_analysis.csv")
        print(f"   - reports/optimal_tilt_analysis.png")
        print("="*70 + "\n")
        
        return results
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
