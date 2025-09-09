import pandas as pd
import os

RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_ninja():
    path = os.path.join(RAW_DIR, "ninja_pv_13.0837_80.2702_uncorrected.csv")
    df = pd.read_csv(path, skiprows=3)  # skip metadata
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").resample("1h").mean(numeric_only=True)
    df.rename(columns={"electricity": "ninja_pv"}, inplace=True)
    return df


def load_nsrdb():
    for fname in os.listdir(RAW_DIR):
        if fname.lower().endswith(".csv") and "nsrdb" in fname.lower():
            path = os.path.join(RAW_DIR, fname)
            print("Loading NSRDB file:", path)

            # NSRDB files usually have 2 header rows
            df = pd.read_csv(path, skiprows=2)

            # Build datetime index
            if "Year" in df.columns and "Month" in df.columns and "Day" in df.columns and "Hour" in df.columns:
                df["time"] = pd.to_datetime(
                    df["Year"].astype(str) + "-" +
                    df["Month"].astype(str) + "-" +
                    # <- remove the dash before time
                    df["Day"].astype(str) + " " +
                    df["Hour"].astype(str) + ":" +
                    df["Minute"].astype(str),
                    format="%Y-%m-%d %H:%M",
                    utc=True
                )
            elif "Date (MM/DD/YYYY)" in df.columns and "Time (HH:MM)" in df.columns:
                df["time"] = pd.to_datetime(
                    df["Date (MM/DD/YYYY)"] + " " + df["Time (HH:MM)"],
                    utc=True
                )

            df = df.set_index("time")

            # Keep only useful columns (adjust as per your file)
            cols = [c for c in df.columns if c in [
                "GHI", "DNI", "DHI", "Temperature", "Wind Speed"]]
            df = df[cols]

            # Resample hourly if needed
            df = df.resample("1h").mean()
            return df
    print("⚠ No NSRDB CSV found in raw data folder")
    return None


def merge_all():
    ninja = load_ninja()
    nsrdb = load_nsrdb()
    if nsrdb is not None:
        df = pd.concat([ninja, nsrdb], axis=1)
    else:
        df = ninja
    outpath = os.path.join(PROCESSED_DIR, "solar_processed.csv")
    df.to_csv(outpath)
    print("✅ Processed dataset saved to:", outpath)


if __name__ == "__main__":
    merge_all()
