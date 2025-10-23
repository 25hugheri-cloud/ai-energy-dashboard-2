import time
import random
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
import plotly.express as px

# --- CONFIG ---
DEVICES = {"hvac_unit": 800, "lighting_floor2": 300, "elevator_motor": 150}
UPDATE_INTERVAL = 2      # seconds between simulated readings
WINDOW_HOURS = 6         # rolling window for display
ANALYZE_EVERY = 60       # seconds between AI analysis
MAX_POINTS = 5000        # keep stream light

# --- STATE ---
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["ts", "device_id", "value"])
    st.session_state.last_analysis = 0
    st.session_state.last_update = 0
    st.session_state.recommendations = []

def simulate_reading(device, base):
    """Simulate power readings with random anomalies."""
    val = base + random.gauss(0, base * 0.1)
    if random.random() < 0.02:
        val *= random.uniform(2, 3)
    return val

def generate_data():
    """Add one reading per device."""
    now = datetime.utcnow()
    new_rows = [{"ts": now, "device_id": d, "value": simulate_reading(d, base)}
                for d, base in DEVICES.items()]
    df_new = pd.DataFrame(new_rows)
    st.session_state.data = pd.concat(
        [st.session_state.data, df_new], ignore_index=True
    ).tail(MAX_POINTS)

def analyze_data():
    """Run Isolation Forest to find anomalies."""
    df = st.session_state.data.copy()
    recs = []
    if df.empty:
        return []
    for device, g in df.groupby("device_id"):
        if len(g) < 20:
            continue
        X = g["value"].values.reshape(-1, 1)
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X)
        g["label"] = model.predict(X)
        anomalies = g[g["label"] == -1]
        frac = len(anomalies) / len(g)
        mean_p = g["value"].mean()
        max_p = g["value"].max()
        recs.append({
            "device": device,
            "mean": mean_p,
            "max": max_p,
            "anomaly_rate": frac
        })
    return recs

# --- UI ---
st.set_page_config("AI Energy Dashboard", layout="wide")
st.title("🏙️ AI Energy Efficiency Dashboard")
st.caption("Live simulation of power monitoring with AI anomaly detection")

# Start/stop simulation
run = st.checkbox("Run simulation", value=True)
placeholder_chart = st.empty()
placeholder_recs = st.empty()

while run:
    # 1. Simulate new data
    generate_data()

    # 2. Analyze every minute
    now = time.time()
    if now - st.session_state.last_analysis > ANALYZE_EVERY:
        st.session_state.recommendations = analyze_data()
        st.session_state.last_analysis = now

    # 3. Plot latest data
    df_display = st.session_state.data.copy()
    cutoff = datetime.utcnow() - timedelta(hours=WINDOW_HOURS)
    df_display = df_display[df_display["ts"] > cutoff]

    fig = px.line(
        df_display, x="ts", y="value", color="device_id",
        title="Live Power Usage", height=400
    )
    fig.update_layout(legend_title_text="Device", xaxis_title="Time (UTC)", yaxis_title="Watts")
    placeholder_chart.plotly_chart(fig, use_container_width=True)

    # 4. Display AI analysis
    if st.session_state.recommendations:
        recs_df = pd.DataFrame(st.session_state.recommendations)
        recs_df["Status"] = recs_df["anomaly_rate"].apply(
            lambda x: "⚠️ Check" if x > 0.08 else "✅ Stable"
        )
        placeholder_recs.dataframe(
            recs_df[["device", "mean", "max", "anomaly_rate", "Status"]],
            use_container_width=True
        )

    # 5. Control simulation speed
    time.sleep(UPDATE_INTERVAL)
