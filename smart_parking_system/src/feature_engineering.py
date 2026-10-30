# ============================================================
#  Smart Parking System – Feature Engineering
#  Extracts meaningful features from the clean dataset.
# ============================================================

import pandas as pd
import numpy as np
import os

CLEAN_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "parking_data_clean.csv")
FEATURE_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "parking_features.csv")


def load_clean(path: str = CLEAN_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"[✓] Loaded clean data → {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ── Time-based features ──────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract granular time features from timestamp."""
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)   # cyclic encoding
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

    # Time-of-day buckets
    def time_bucket(h):
        if   0  <= h <  6: return 0   # Night
        elif 6  <= h < 12: return 1   # Morning
        elif 12 <= h < 18: return 2   # Afternoon
        else:              return 3   # Evening

    df["time_of_day"] = df["hour"].apply(time_bucket)

    # Peak hours flag (9–17 weekdays)
    df["is_peak_hour"] = (
        (df["hour"] >= 9) & (df["hour"] <= 17) & (df["is_weekend"] == 0)
    ).astype(int)

    print("[✓] Time features added")
    return df


# ── Historical usage features ────────────────────────────────
def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each slot compute:
      - Rolling 1-hour occupancy rate  (last 12 readings)
      - Rolling 3-hour occupancy rate  (last 36 readings)
      - Previous reading occupancy
    """
    df = df.sort_values(["slot_id", "timestamp"]).reset_index(drop=True)

    df["prev_occupied"] = df.groupby("slot_id")["occupied"].shift(1).fillna(0)

    df["rolling_1h_rate"] = (
        df.groupby("slot_id")["occupied"]
          .transform(lambda x: x.shift(1).rolling(12,  min_periods=1).mean())
          .fillna(0)
    )
    df["rolling_3h_rate"] = (
        df.groupby("slot_id")["occupied"]
          .transform(lambda x: x.shift(1).rolling(36,  min_periods=1).mean())
          .fillna(0)
    )

    print("[✓] Historical usage features added")
    return df


# ── Lot-level features ───────────────────────────────────────
def add_lot_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregated lot-level occupancy at each timestamp.
    """
    lot_occ = (
        df.groupby("timestamp")["occupied"]
          .mean()
          .rename("lot_occupancy_rate")
    )
    df = df.merge(lot_occ, on="timestamp", how="left")
    df["lot_available_slots"] = (
        (1 - df["lot_occupancy_rate"]) * df["slot_id"].nunique()
    ).round().astype(int)

    print("[✓] Lot-level features added")
    return df


# ── Distance-based features ──────────────────────────────────
def add_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise distance and add a confidence score.
    """
    df["distance_norm"] = (df["distance_cm"] - df["distance_cm"].min()) / \
                          (df["distance_cm"].max() - df["distance_cm"].min())

    # Confidence: how far reading is from 50 cm threshold
    df["sensor_confidence"] = np.abs(df["distance_cm"] - 50) / 200

    print("[✓] Sensor features added")
    return df


# ── Main ─────────────────────────────────────────────────────
def engineer_features(save: bool = True) -> pd.DataFrame:
    df = load_clean()
    df = add_time_features(df)
    df = add_historical_features(df)
    df = add_lot_features(df)
    df = add_sensor_features(df)

    print(f"\n[Summary] Feature dataset shape : {df.shape}")

    if save:
        df.to_csv(FEATURE_PATH, index=False)
        print(f"[✓] Feature data saved → {FEATURE_PATH}")

    return df


# ── Feature list used for ML ─────────────────────────────────
FEATURE_COLUMNS = [
    "hour", "minute", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "time_of_day", "is_peak_hour",
    "slot_id", "distance_cm", "distance_norm", "sensor_confidence",
    "prev_occupied", "rolling_1h_rate", "rolling_3h_rate",
    "lot_occupancy_rate", "lot_available_slots",
    "day_encoded",
]
TARGET_COLUMN = "occupied"


if __name__ == "__main__":
    engineer_features()
