# Solar PV Placement - PVLib Migration Summary

## Changes Made to Use PVLib Instead of Ninja PV Dataset

### 1. Updated pvlib_simulation.py
- Enhanced the PVLib simulation to properly handle NSRDB data
- Added function `load_and_simulate_pvlib()` with configurable parameters
- Properly handles DHI estimation when not available in NSRDB
- Scales results based on number of panels
- Outputs: AC power, DC power, energy in kWh

### 2. Updated data_processing.py
- Replaced `load_ninja()` function with `load_pvlib_data()`
- Uses existing pvlib results or runs simulation if needed
- Merges PVLib energy results with NSRDB weather data
- Creates unified dataset in solar_processed.csv

### 3. Updated energy_simulation.py
- Modified to use `energy_kwh` from PVLib simulation instead of manual GHI calculation
- Scales energy based on number of panels (with proper reference to base simulation)
- Fallback to manual GHI calculation if PVLib data not available
- More accurate physics-based energy estimation

### 4. Updated panel_placement.py
- Uses `energy_kwh` or `ac_power` from PVLib simulation
- Fallback to GHI data if PVLib results not available
- More realistic solar potential calculations

## Results
✅ Successfully generated 8760 hourly records for full year 2014
✅ PVLib simulation accounts for:
   - Solar geometry and sun position
   - Panel tilt and azimuth optimization
   - Temperature effects on panels
   - Spectral and AOI losses
   - Inverter efficiency

✅ Annual energy: ~4,761 kWh for 20 panels (238 kWh per panel)
✅ All visualization and analysis scripts now use physics-based PVLib data

## Data Flow
NSRDB (raw weather) → PVLib simulation → solar_processed.csv → analysis scripts

This provides much more accurate solar energy estimates compared to the simplified ninja PV dataset.
