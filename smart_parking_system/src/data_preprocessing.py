# ============================================================
#  Smart Parking System – Data Preprocessing
#  Cleans, validates and prepares the raw dataset for ML.
# ============================================================

import pandas as pd
import numpy as np
import os

RAW_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "parking_data.csv")
CLEAN_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "parking_data_clean.csv")


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Load raw CSV produced by generate_dataset.py."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"[✓] Loaded raw data  → {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["timestamp", "slot_id"])
    removed = before - len(df)
    print(f"[✓] Duplicates removed : {removed}")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum().sum()
    if missing > 0:
        # Fill numeric with median, categorical with mode
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        print(f"[✓] Filled {missing} missing values")
    else:
        print("[✓] No missing values found")
    return df


def validate_sensor_readings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove physically impossible sensor readings.
    Ultrasonic range: 2 cm – 400 cm.
    """
    before = len(df)
    df = df[(df["distance_cm"] >= 2) & (df["distance_cm"] <= 400)]
    removed = before - len(df)
    print(f"[✓] Invalid sensor readings removed : {removed}")
    return df


def add_occupancy_label(df: pd.DataFrame,
                        threshold_cm: float = 50.0) -> pd.DataFrame:
    """
    Re-derive occupancy label from distance (safety check).
    distance < threshold  →  occupied  (1)
    distance >= threshold →  vacant    (0)
    """
    df["occupied_derived"] = (df["distance_cm"] < threshold_cm).astype(int)

    # Flag mismatches between sensor label and derived label
    mismatch = (df["occupied"] != df["occupied_derived"]).sum()
    print(f"[✓] Label mismatches detected : {mismatch} (using original labels)")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode day_name for later use."""
    day_order = ["Monday", "Tuesday", "Wednesday",
                 "Thursday", "Friday", "Saturday", "Sunday"]
    df["day_encoded"] = df["day_name"].apply(
        lambda x: day_order.index(x) if x in day_order else -1
    )
    return df


def preprocess(save: bool = True) -> pd.DataFrame:
    df = load_raw()
    df = remove_duplicates(df)
    df = handle_missing(df)
    df = validate_sensor_readings(df)
    df = add_occupancy_label(df)
    df = encode_categoricals(df)

    print(f"\n[Summary] Clean dataset shape : {df.shape}")
    print(df.dtypes)

    if save:
        df.to_csv(CLEAN_PATH, index=False)
        print(f"\n[✓] Clean data saved → {CLEAN_PATH}")

    return df


if __name__ == "__main__":
    preprocess()
