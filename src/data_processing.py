import pandas as pd
import os
import sys
from pvlib_simulation import load_and_simulate_pvlib

# Setup absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Ensure directories exist
os.makedirs(DATA_PROCESSED, exist_ok=True)


def load_pvlib_data():
    """Load already generated PVLib simulation results"""
    pvlib_path = os.path.join(DATA_PROCESSED, "pvlib_results.csv")
    
    if os.path.exists(pvlib_path):
        print(f"📂 Loading existing PVLib results from {pvlib_path}")
        df = pd.read_csv(pvlib_path)
        df["time"] = pd.to_datetime(df["time"])
        print(f"✅ PVLib data loaded - {len(df)} records")
        return df
    else:
        print("🔋 Running PVLib simulation...")
        results = load_and_simulate_pvlib(tilt_angle=20, azimuth_angle=180, num_panels=10)
        
        if results is not None:
            # Reset index to get time as a column
            df = results.reset_index()
            df.rename(columns={"index": "time"}, inplace=True)
            
            # Save results
            pvlib_path = os.path.join(DATA_PROCESSED, "pvlib_results.csv")
            df.to_csv(pvlib_path, index=False)
            
            print(f"✅ PVLib simulation complete - {len(df)} records")
            return df
        else:
            print("❌ PVLib simulation failed")
            return None


def load_nsrdb():
    path = os.path.join(DATA_RAW, "nsrdb.csv")
    print(f"Loading NSRDB file: {path}")

    df = pd.read_csv(path, skiprows=2)  # Skip the 2 header rows before actual data

    # Create datetime column (timezone-naive to match PVLib)
    df["time"] = pd.to_datetime(
        df["Year"].astype(str) + "-" +
        df["Month"].astype(str) + "-" +
        df["Day"].astype(str) + " " +
        df["Hour"].astype(str) + ":" +
        df["Minute"].astype(str)
        # Removed utc=True to keep timezone-naive
    )

    # Keep only useful columns
    df = df[[
        "time", "DHI", "DNI", "GHI",
        "Temperature", "Pressure",
        "Relative Humidity", "Wind Speed"
    ]]

    # Resample to hourly average (using 'h' instead of deprecated 'H')
    df = df.set_index("time").resample("1h").mean().reset_index()

    return df



def merge_all():
    """Merge PVLib simulation results with NSRDB weather data"""
    pvlib_data = load_pvlib_data()
    nsrdb = load_nsrdb()
    
    if pvlib_data is not None and nsrdb is not None:
        # Ensure both datasets have time as datetime and handle timezone consistency
        pvlib_data["time"] = pd.to_datetime(pvlib_data["time"])
        nsrdb["time"] = pd.to_datetime(nsrdb["time"])
        
        # Make both timezone-naive for consistent merging
        if pvlib_data["time"].dt.tz is not None:
            pvlib_data["time"] = pvlib_data["time"].dt.tz_localize(None)
        if nsrdb["time"].dt.tz is not None:
            nsrdb["time"] = nsrdb["time"].dt.tz_localize(None)
        
        pvlib_data = pvlib_data.set_index("time")
        nsrdb = nsrdb.set_index("time")
        
        # Merge on time index
        df = pd.merge(pvlib_data, nsrdb, left_index=True, right_index=True, how="inner")
        df = df.reset_index()  # Reset index to get time as column again
        
        print(f"✅ Successfully merged datasets")
        print(f"   PVLib records: {len(pvlib_data)}")
        print(f"   NSRDB records: {len(nsrdb)}")
        print(f"   Merged records: {len(df)}")
        
    elif pvlib_data is not None:
        df = pvlib_data
        print("⚠️  Using only PVLib data (NSRDB not available)")
    else:
        print("❌ No data available for processing")
        return
    
    outpath = os.path.join(DATA_PROCESSED, "solar_processed.csv")
    df.to_csv(outpath, index=False)
    print("✅ Processed dataset saved to:", outpath)
    print(f"   Total records: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    if len(df) > 0:
        print(f"   Date range: {df['time'].min()} to {df['time'].max()}")


if __name__ == "__main__":
    merge_all()
