# ============================================================
#  Smart Parking System – Dataset Generator
#  Simulates ultrasonic sensor readings for 10 parking slots
#  over 30 days (5-minute intervals) and saves to CSV.
# ============================================================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ── Configuration ────────────────────────────────────────────
NUM_SLOTS      = 10          # number of parking slots
DAYS           = 30          # days of data to simulate
INTERVAL_MIN   = 5           # sensor poll interval in minutes
RANDOM_SEED    = 42
OUTPUT_PATH    = os.path.join(os.path.dirname(__file__), "parking_data.csv")

np.random.seed(RANDOM_SEED)

# ── Occupancy probability model ──────────────────────────────
#   Varies by hour-of-day and day-of-week to be realistic.
def occupancy_probability(hour: int, day_of_week: int) -> float:
    """
    Returns the probability that a slot is occupied at a given
    hour and day.  Weekdays are busier than weekends; peak hours
    are 9-17.
    """
    is_weekend = day_of_week >= 5          # Sat=5, Sun=6

    # Base probability curves
    if 9 <= hour <= 17:
        base = 0.80 if not is_weekend else 0.40
    elif 7 <= hour <= 9 or 17 <= hour <= 19:
        base = 0.60 if not is_weekend else 0.30
    elif 19 <= hour <= 22:
        base = 0.40 if not is_weekend else 0.25
    else:                                  # night / early morning
        base = 0.10

    # Add slight random variation per call
    return float(np.clip(base + np.random.uniform(-0.05, 0.05), 0.0, 1.0))


# ── Sensor reading model ─────────────────────────────────────
def ultrasonic_distance(occupied: bool) -> float:
    """
    Simulate ultrasonic sensor distance (cm).
    Occupied slot  → short distance  (5 – 30 cm)
    Vacant  slot   → long  distance  (80 – 200 cm)
    """
    if occupied:
        return round(np.random.uniform(5, 30), 2)
    else:
        return round(np.random.uniform(80, 200), 2)


# ── Main generation loop ─────────────────────────────────────
def generate_dataset() -> pd.DataFrame:
    records = []
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    total_intervals = (DAYS * 24 * 60) // INTERVAL_MIN

    for i in range(total_intervals):
        ts  = start_time + timedelta(minutes=i * INTERVAL_MIN)
        hour        = ts.hour
        day_of_week = ts.weekday()          # 0=Mon … 6=Sun
        date_str    = ts.strftime("%Y-%m-%d")
        time_str    = ts.strftime("%H:%M:%S")

        for slot_id in range(1, NUM_SLOTS + 1):
            prob     = occupancy_probability(hour, day_of_week)
            occupied = int(np.random.rand() < prob)
            distance = ultrasonic_distance(bool(occupied))

            records.append({
                "timestamp"   : ts,
                "date"        : date_str,
                "time"        : time_str,
                "hour"        : hour,
                "minute"      : ts.minute,
                "day_of_week" : day_of_week,       # 0=Mon
                "day_name"    : ts.strftime("%A"),
                "is_weekend"  : int(day_of_week >= 5),
                "slot_id"     : slot_id,
                "distance_cm" : distance,
                "occupied"    : occupied,           # 1=occupied, 0=vacant
            })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[✓] Dataset saved  → {OUTPUT_PATH}")
    print(f"    Rows : {len(df):,}")
    print(f"    Cols : {list(df.columns)}")
    return df


if __name__ == "__main__":
    generate_dataset()
