# ============================================================
#  Smart Parking System – EDA & Visualisation
#  Generates all exploratory analysis plots saved to /reports
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CLEAN_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "parking_data_clean.csv")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


def load_data(path: str = CLEAN_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


# ── 1. Overall occupancy distribution ───────────────────────
def plot_occupancy_distribution(df: pd.DataFrame):
    counts = df["occupied"].value_counts()
    labels = ["Vacant", "Occupied"]
    colors = ["#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Pie chart
    axes[0].pie(counts, labels=labels, autopct="%1.1f%%",
                colors=colors, startangle=90)
    axes[0].set_title("Overall Slot Occupancy")

    # Bar chart
    axes[1].bar(labels, counts.values, color=colors)
    axes[1].set_ylabel("Count")
    axes[1].set_title("Occupied vs Vacant Readings")

    plt.tight_layout()
    _save("01_occupancy_distribution.png")


# ── 2. Hourly occupancy rate ─────────────────────────────────
def plot_hourly_occupancy(df: pd.DataFrame):
    hourly = df.groupby("hour")["occupied"].mean().reset_index()
    hourly.columns = ["Hour", "Occupancy Rate"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(hourly["Hour"], hourly["Occupancy Rate"],
                    alpha=0.3, color="#4C72B0")
    ax.plot(hourly["Hour"], hourly["Occupancy Rate"],
            marker="o", color="#4C72B0", linewidth=2)
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Avg Occupancy Rate")
    ax.set_title("Hourly Average Parking Occupancy")
    ax.axhline(0.5, color="red", linestyle="--", label="50% line")
    ax.legend()
    plt.tight_layout()
    _save("02_hourly_occupancy.png")


# ── 3. Day-of-week occupancy ─────────────────────────────────
def plot_daily_occupancy(df: pd.DataFrame):
    day_order = ["Monday","Tuesday","Wednesday",
                 "Thursday","Friday","Saturday","Sunday"]
    daily = (df.groupby("day_name")["occupied"]
               .mean()
               .reindex(day_order)
               .reset_index())
    daily.columns = ["Day", "Occupancy Rate"]

    fig, ax = plt.subplots(figsize=(9, 4))
    colors  = ["#C44E52" if d in ["Saturday","Sunday"] else "#4C72B0"
               for d in daily["Day"]]
    ax.bar(daily["Day"], daily["Occupancy Rate"], color=colors)
    ax.set_ylabel("Avg Occupancy Rate")
    ax.set_title("Day-of-Week Parking Occupancy")
    ax.set_xticklabels(daily["Day"], rotation=30)
    plt.tight_layout()
    _save("03_daily_occupancy.png")


# ── 4. Per-slot occupancy ────────────────────────────────────
def plot_slot_occupancy(df: pd.DataFrame):
    slot = df.groupby("slot_id")["occupied"].mean().reset_index()
    slot.columns = ["Slot", "Occupancy Rate"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(slot["Slot"].astype(str), slot["Occupancy Rate"], color="#8172B2")
    ax.set_xlabel("Slot ID")
    ax.set_ylabel("Avg Occupancy Rate")
    ax.set_title("Average Occupancy per Parking Slot")
    plt.tight_layout()
    _save("04_slot_occupancy.png")


# ── 5. Heatmap – hour vs day ─────────────────────────────────
def plot_heatmap(df: pd.DataFrame):
    pivot = df.pivot_table(
        values="occupied", index="day_name", columns="hour", aggfunc="mean"
    )
    day_order = ["Monday","Tuesday","Wednesday",
                 "Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex(day_order)

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3,
                cbar_kws={"label": "Occupancy Rate"})
    ax.set_title("Occupancy Heatmap – Hour vs Day of Week")
    ax.set_xlabel("Hour")
    ax.set_ylabel("")
    plt.tight_layout()
    _save("05_heatmap_hour_day.png")


# ── 6. Distance distribution ─────────────────────────────────
def plot_distance_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, color, name in [(0, "#55A868", "Vacant"),
                                (1, "#C44E52", "Occupied")]:
        subset = df[df["occupied"] == label]["distance_cm"]
        ax.hist(subset, bins=60, alpha=0.6, color=color, label=name)
    ax.axvline(50, color="black", linestyle="--", label="50 cm threshold")
    ax.set_xlabel("Distance (cm)")
    ax.set_ylabel("Frequency")
    ax.set_title("Ultrasonic Sensor Distance Distribution")
    ax.legend()
    plt.tight_layout()
    _save("06_distance_distribution.png")


# ── 7. Rolling occupancy trend (lot level) ───────────────────
def plot_rolling_trend(df: pd.DataFrame):
    lot = (df.groupby("timestamp")["occupied"]
             .mean()
             .reset_index()
             .sort_values("timestamp"))
    lot["rolling_24h"] = lot["occupied"].rolling(window=288, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(lot["timestamp"], lot["occupied"],
            alpha=0.2, color="#4C72B0", linewidth=0.5, label="Raw")
    ax.plot(lot["timestamp"], lot["rolling_24h"],
            color="#C44E52", linewidth=2, label="24-hour Rolling Avg")
    ax.set_xlabel("Date")
    ax.set_ylabel("Lot Occupancy Rate")
    ax.set_title("Parking Lot Occupancy Over Time")
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    _save("07_rolling_trend.png")


# ── 8. Correlation matrix ────────────────────────────────────
def plot_correlation(df: pd.DataFrame):
    num_cols = ["hour", "minute", "day_of_week", "is_weekend",
                "distance_cm", "occupied", "is_peak_hour"]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    _save("08_correlation_matrix.png")


# ── Helper ───────────────────────────────────────────────────
def _save(filename: str):
    path = os.path.join(REPORTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Saved → reports/{filename}")


# ── Run all EDA ───────────────────────────────────────────────
def run_eda():
    print("=== Exploratory Data Analysis ===\n")
    df = load_data()

    print(f"Dataset shape   : {df.shape}")
    print(f"Date range      : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Slots           : {df['slot_id'].nunique()}")
    print(f"Occupancy rate  : {df['occupied'].mean()*100:.1f}%\n")

    plot_occupancy_distribution(df)
    plot_hourly_occupancy(df)
    plot_daily_occupancy(df)
    plot_slot_occupancy(df)
    plot_heatmap(df)
    plot_distance_distribution(df)
    plot_rolling_trend(df)
    plot_correlation(df)

    print("\n[✓] All EDA plots saved to reports/")


if __name__ == "__main__":
    run_eda()
