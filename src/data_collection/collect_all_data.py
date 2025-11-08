"""
Master script to collect and prepare all datasets for ML training
Run this first to set up all required data
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*70)
print(" 📦 SOLAR PV + HYDROGEN STORAGE: DATA COLLECTION PIPELINE")
print("="*70)

# Step 1: Generate/Load Solar Data
print("\n[STEP 1/5] Collecting Solar Power Generation Data...")
try:
    from data_collection.fetch_solar_data import load_and_process_solar_data, save_processed_data
    solar_df = load_and_process_solar_data()
    solar_path = save_processed_data(solar_df)
    print(f"✅ Solar data ready: {len(solar_df)} records")
except Exception as e:
    print(f"❌ Error in solar data: {e}")
    solar_df = None

# Step 2: Generate Load Profile
print("\n[STEP 2/5] Generating Residential Load Profile...")
try:
    from data_collection.generate_load_profile import generate_residential_load, save_load_profile
    load_df = generate_residential_load(
        num_days=365,
        avg_daily_kwh=30,
        min_daily_kwh=20,
        max_daily_kwh=40,
        peak_kw=8
    )
    load_path = save_load_profile(load_df)
    print(f"✅ Load profile ready: {len(load_df)} records")
except Exception as e:
    print(f"❌ Error in load data: {e}")
    load_df = None

# Step 3: Create Hydrogen Equipment Database
print("\n[STEP 3/5] Creating Hydrogen Equipment Specifications...")
try:
    from data_collection.hydrogen_equipment_specs import create_hydrogen_equipment_database, save_equipment_database
    h2_db = create_hydrogen_equipment_database()
    save_equipment_database(h2_db)
    print(f"✅ Hydrogen equipment database ready: {len(h2_db)} tables")
except Exception as e:
    print(f"❌ Error in hydrogen data: {e}")
    h2_db = None

# Step 4: Merge and Create Master Dataset
print("\n[STEP 4/5] Creating Master Dataset...")
try:
    if solar_df is not None and load_df is not None:
        # Align timestamps
        solar_df['time'] = pd.to_datetime(solar_df['time'])
        load_df['time'] = pd.to_datetime(load_df['time'])
        
        # Merge on time
        master_df = pd.merge(solar_df, load_df, on='time', how='inner')
        
        # Calculate net load (consumption - generation)
        master_df['net_load'] = master_df['load_kwh'] - master_df['energy_kwh']
        
        # Add time features for ML
        master_df['hour'] = master_df['time'].dt.hour
        master_df['day_of_week'] = master_df['time'].dt.dayofweek
        master_df['month'] = master_df['time'].dt.month
        master_df['day_of_year'] = master_df['time'].dt.dayofyear
        master_df['is_weekend'] = (master_df['day_of_week'] >= 5).astype(int)
        
        # Add sin/cos encoding for cyclical features
        master_df['hour_sin'] = np.sin(2 * np.pi * master_df['hour'] / 24)
        master_df['hour_cos'] = np.cos(2 * np.pi * master_df['hour'] / 24)
        master_df['month_sin'] = np.sin(2 * np.pi * master_df['month'] / 12)
        master_df['month_cos'] = np.cos(2 * np.pi * master_df['month'] / 12)
        
        # Save master dataset
        os.makedirs("../data/processed/", exist_ok=True)
        master_path = "../data/processed/master_dataset.csv"
        master_df.to_csv(master_path, index=False)
        
        print(f"✅ Master dataset created: {master_path}")
        print(f"   Shape: {master_df.shape}")
        print(f"   Columns: {list(master_df.columns)}")
        
    else:
        print("⚠️  Skipping master dataset creation - missing source data")
        master_df = None
        
except Exception as e:
    print(f"❌ Error creating master dataset: {e}")
    master_df = None

# Step 5: Create Train/Test Splits
print("\n[STEP 5/5] Creating Train/Test/Validation Splits...")
try:
    if master_df is not None:
        # Chronological split (important for time series)
        train_size = int(len(master_df) * 0.7)
        val_size = int(len(master_df) * 0.15)
        
        train_df = master_df.iloc[:train_size]
        val_df = master_df.iloc[train_size:train_size+val_size]
        test_df = master_df.iloc[train_size+val_size:]
        
        # Save splits
        splits_dir = "../data/processed/splits/"
        os.makedirs(splits_dir, exist_ok=True)
        
        train_df.to_csv(splits_dir + "train.csv", index=False)
        val_df.to_csv(splits_dir + "val.csv", index=False)
        test_df.to_csv(splits_dir + "test.csv", index=False)
        
        print(f"✅ Data splits created:")
        print(f"   Train: {len(train_df)} records ({len(train_df)/len(master_df):.1%})")
        print(f"   Val:   {len(val_df)} records ({len(val_df)/len(master_df):.1%})")
        print(f"   Test:  {len(test_df)} records ({len(test_df)/len(master_df):.1%})")
    else:
        print("⚠️  Skipping train/test split - no master dataset")
        
except Exception as e:
    print(f"❌ Error creating splits: {e}")

# Summary
print("\n" + "="*70)
print(" 📊 DATA COLLECTION SUMMARY")
print("="*70)

if master_df is not None:
    print(f"\n✅ ALL DATA READY FOR ML TRAINING")
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total records: {len(master_df):,}")
    print(f"   Date range: {master_df['time'].min()} to {master_df['time'].max()}")
    print(f"   Solar generation: {master_df['energy_kwh'].sum():,.0f} kWh/year")
    print(f"   Load consumption: {master_df['load_kwh'].sum():,.0f} kWh/year")
    print(f"   Net energy balance: {master_df['net_load'].sum():,.0f} kWh/year")
    print(f"   Self-sufficiency: {(1 - master_df[master_df['net_load'] > 0]['net_load'].sum() / master_df['load_kwh'].sum()):.1%}")
    
    print(f"\n📁 Output Files:")
    print(f"   Master dataset: ../data/processed/master_dataset.csv")
    print(f"   Train/val/test: ../data/processed/splits/")
    print(f"   H2 equipment: ../data/raw/hydrogen_*.csv")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Run ML model training: python ml_models/train_all_models.py")
    print(f"   2. Run optimization: python optimization/run_optimization.py")
    print(f"   3. Evaluate results: python evaluation/generate_reports.py")
else:
    print("\n⚠️  DATA COLLECTION INCOMPLETE")
    print("   Please check error messages above and retry")

print("\n" + "="*70)
