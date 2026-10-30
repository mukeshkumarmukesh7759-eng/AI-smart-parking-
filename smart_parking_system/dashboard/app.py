# ============================================================
#  Smart Parking System – Streamlit Dashboard
#  Run with:  streamlit run dashboard/app.py
# ============================================================

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Path setup ───────────────────────────────────────────────
ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC    = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
sys.path.insert(0, ROOT)

from src.sensor_simulator   import ParkingLotSimulator
from src.feature_engineering import FEATURE_COLUMNS

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title = "Smart Parking System",
    page_icon  = "🅿️",
    layout     = "wide",
)

# ── Hide Streamlit deploy button ─────────────────────────────
st.markdown(
    "<style>[data-testid='stAppDeployButton'] { display: none; }</style>",
    unsafe_allow_html=True,
)

# ── Constants ────────────────────────────────────────────────
NUM_SLOTS   = 10
MODEL_PATH  = os.path.join(ROOT, "models", "best_model.pkl")
FEATURE_CSV = os.path.join(ROOT, "data",   "parking_features.csv")
METRICS_CSV = os.path.join(ROOT, "reports","model_metrics.csv")

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

# ── Load historical data ──────────────────────────────────────
@st.cache_data
def load_historical():
    if os.path.exists(FEATURE_CSV):
        df = pd.read_csv(FEATURE_CSV, parse_dates=["timestamp"])
        return df
    return pd.DataFrame()

# ── Session state ─────────────────────────────────────────────
if "simulator"  not in st.session_state:
    st.session_state.simulator   = ParkingLotSimulator(NUM_SLOTS)
if "history"    not in st.session_state:
    st.session_state.history     = []
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False

# ── Helpers ───────────────────────────────────────────────────
def slot_color(occupied: bool) -> str:
    return "#e74c3c" if occupied else "#2ecc71"

def predict_occupancy(model, slot_readings: list) -> list:
    """Run ML prediction on current sensor readings."""
    if model is None:
        return [r["occupied"] for r in slot_readings]

    hour        = datetime.now().hour
    minute      = datetime.now().minute
    day_of_week = datetime.now().weekday()
    is_weekend  = int(day_of_week >= 5)
    is_peak     = int(9 <= hour <= 17 and not is_weekend)
    lot_occ     = np.mean([r["occupied"] for r in slot_readings])
    lot_avail   = int((1 - lot_occ) * NUM_SLOTS)

    predictions = []
    for r in slot_readings:
        dist = r["distance_cm"]
        row  = {
            "hour"               : hour,
            "minute"             : minute,
            "day_of_week"        : day_of_week,
            "is_weekend"         : is_weekend,
            "hour_sin"           : np.sin(2 * np.pi * hour / 24),
            "hour_cos"           : np.cos(2 * np.pi * hour / 24),
            "minute_sin"         : np.sin(2 * np.pi * minute / 60),
            "minute_cos"         : np.cos(2 * np.pi * minute / 60),
            "time_of_day"        : 0 if hour<6 else (1 if hour<12 else (2 if hour<18 else 3)),
            "is_peak_hour"       : is_peak,
            "slot_id"            : r["slot_id"],
            "distance_cm"        : dist,
            "distance_norm"      : np.clip((dist - 2) / 398, 0, 1),
            "sensor_confidence"  : abs(dist - 50) / 200,
            "prev_occupied"      : r["occupied"],
            "rolling_1h_rate"    : lot_occ,
            "rolling_3h_rate"    : lot_occ,
            "lot_occupancy_rate" : lot_occ,
            "lot_available_slots": lot_avail,
            "day_encoded"        : day_of_week,
        }
        X = pd.DataFrame([row])[FEATURE_COLUMNS]
        predictions.append(int(model.predict(X)[0]))
    return predictions

# ── UI ────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;'>🅿️ Smart Parking System</h1>"
    "<p style='text-align:center; color:gray;'>AI + Ultrasonic Sensor Based Parking Monitor</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    refresh = st.button("🔄 Refresh Sensor Readings", use_container_width=True)
    auto    = st.checkbox("⏱️ Auto-refresh (every 5 s)", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto
    st.markdown("---")
    use_ml  = st.checkbox("🤖 Use ML Predictions", value=True)
    st.markdown("---")
    st.subheader("📊 Model Metrics")
    if os.path.exists(METRICS_CSV):
        metrics_df = pd.read_csv(METRICS_CSV)
        st.dataframe(metrics_df.set_index("Model"), use_container_width=True)
    else:
        st.info("Run main.py to train the model first.")

# Read sensors
model    = load_model()
readings = st.session_state.simulator.read_all_slots()

if use_ml and model:
    preds = predict_occupancy(model, readings)
    for i, r in enumerate(readings):
        r["occupied"] = preds[i]
        r["status"]   = "Occupied" if preds[i] else "Vacant"

# Store history
ts = datetime.now().strftime("%H:%M:%S")
occ_count = sum(r["occupied"] for r in readings)
st.session_state.history.append({"time": ts, "occupied": occ_count,
                                  "vacant": NUM_SLOTS - occ_count})
if len(st.session_state.history) > 50:
    st.session_state.history.pop(0)

# Summary metrics
total    = NUM_SLOTS
occupied = occ_count
vacant   = total - occupied
occ_pct  = occupied / total * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("🅿️ Total Slots",    total)
col2.metric("🔴 Occupied",        occupied, delta=None)
col3.metric("🟢 Available",       vacant,   delta=None)
col4.metric("📊 Occupancy",       f"{occ_pct:.1f}%")

st.markdown("---")

# ── Parking lot grid ──────────────────────────────────────────
st.subheader("🏢 Real-Time Parking Slot Status")
COLS = 5
slot_cols = st.columns(COLS)
for i, r in enumerate(readings):
    col   = slot_cols[i % COLS]
    color = slot_color(bool(r["occupied"]))
    icon  = "🔴" if r["occupied"] else "🟢"
    col.markdown(
        f"""<div style='background:{color}22; border:2px solid {color};
         border-radius:12px; padding:12px; text-align:center; margin:4px;'>
        <b>Slot {r['slot_id']}</b><br/>
        {icon} <b>{r['status']}</b><br/>
        <small>{r['distance_cm']} cm</small>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

# Gauge chart
with chart_col1:
    st.subheader("Occupancy Gauge")
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = occ_pct,
        delta = {"reference": 50},
        gauge = {
            "axis" : {"range": [0, 100]},
            "bar"  : {"color": "#e74c3c" if occ_pct > 70 else "#2ecc71"},
            "steps": [
                {"range": [0,  50], "color": "#d5f5e3"},
                {"range": [50, 80], "color": "#fdebd0"},
                {"range": [80,100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line" : {"color": "red", "width": 4},
                "thickness": 0.75, "value": 80
            },
        },
        title = {"text": "Lot Occupancy (%)"},
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# Live history line chart
with chart_col2:
    st.subheader("Live Occupancy Trend")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        fig2    = px.line(hist_df, x="time", y=["occupied","vacant"],
                          labels={"value":"Slots","time":"Time"},
                          color_discrete_map={"occupied":"#e74c3c","vacant":"#2ecc71"})
        fig2.update_layout(height=300, legend_title="Status")
        st.plotly_chart(fig2, use_container_width=True)

# ── Historical analysis ───────────────────────────────────────
hist_data = load_historical()
if not hist_data.empty:
    st.markdown("---")
    st.subheader("📈 Historical Analysis")
    tab1, tab2, tab3 = st.tabs(["Hourly", "Day of Week", "Slot Breakdown"])

    with tab1:
        hourly = hist_data.groupby("hour")["occupied"].mean().reset_index()
        fig3   = px.bar(hourly, x="hour", y="occupied",
                        labels={"hour":"Hour","occupied":"Avg Occupancy Rate"},
                        color="occupied", color_continuous_scale="RdYlGn_r")
        fig3.update_layout(title="Average Occupancy by Hour")
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        day_order = ["Monday","Tuesday","Wednesday",
                     "Thursday","Friday","Saturday","Sunday"]
        daily = (hist_data.groupby("day_name")["occupied"]
                           .mean()
                           .reindex(day_order)
                           .reset_index())
        fig4  = px.bar(daily, x="day_name", y="occupied",
                       labels={"day_name":"Day","occupied":"Avg Occupancy Rate"},
                       color="occupied", color_continuous_scale="RdYlGn_r")
        fig4.update_layout(title="Average Occupancy by Day of Week")
        st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        slot_occ = hist_data.groupby("slot_id")["occupied"].mean().reset_index()
        fig5 = px.bar(slot_occ, x="slot_id", y="occupied",
                      labels={"slot_id":"Slot ID","occupied":"Avg Occupancy"},
                      color="occupied", color_continuous_scale="RdYlGn_r")
        fig5.update_layout(title="Per-Slot Average Occupancy")
        st.plotly_chart(fig5, use_container_width=True)

# ── Raw readings table ────────────────────────────────────────
with st.expander("📋 Raw Sensor Readings"):
    df_raw = pd.DataFrame(readings)[["slot_id","distance_cm","status","timestamp"]]
    st.dataframe(df_raw, use_container_width=True)

# ── Auto-refresh ──────────────────────────────────────────────
if st.session_state.auto_refresh:
    time.sleep(5)
    st.rerun()

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
           f"Model: {'Loaded ✓' if model else 'Not found – run main.py first'}")
