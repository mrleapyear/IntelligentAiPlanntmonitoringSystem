"""
Streamlit Web Dashboard for PVDF Plant Health Monitoring
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import joblib
import os
import serial
import threading
import queue
import json
from collections import deque

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="🌱 PVDF Plant Health Monitor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1B5E20, #4CAF50);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.status-card {
    padding: 1.5rem;
    border-radius: 12px;
    color: white;
    text-align: center;
}
.healthy-card { background: linear-gradient(135deg, #43A047, #66BB6A); }
.pest-card { background: linear-gradient(135deg, #F57C00, #FF9800); }
.water-card { background: linear-gradient(135deg, #1976D2, #2196F3); }
.unknown-card { background: linear-gradient(135deg, #757575, #9E9E9E); }
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
def init_session_state():
    defaults = {
        'serial_connected': False,
        'serial_port': 'COM3',
        'baud_rate': 115200,
        'data_source': 'Simulated',
        'data_queue': queue.Queue(maxsize=1000),
        'history': deque(maxlen=1000),
        'alerts': deque(maxlen=50),
        'current_status': 'UNKNOWN',
        'health_score': 50.0,
        'last_update': None,
        'model_loaded': False,
        'ai_model': None,
        'scaler': None,
        'features_history': deque(maxlen=100),
        'monitoring_active': True,
        'auto_refresh': True,
        'refresh_interval': 5
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ============================================
# SIMULATED DATA
# ============================================
def generate_simulated_data():
    status = np.random.choice(
        ["HEALTHY", "PEST STRESS", "WATER STRESS"],
        p=[0.6, 0.2, 0.2]
    )
    if status == "HEALTHY":
        features = np.random.uniform(20, 60, 8)
        score = np.random.uniform(80, 100)
    elif status == "PEST STRESS":
        features = np.random.uniform(60, 120, 8)
        score = np.random.uniform(40, 70)
    else:
        features = np.random.uniform(5, 30, 8)
        score = np.random.uniform(30, 55)

    return {
        "timestamp": datetime.now(),
        "features": features.tolist(),
        "status": status,
        "health_score": score
    }

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================
def create_health_gauge(score, status):
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Plant Health Score"},
        gauge={"axis": {"range": [0, 100]}}
    ))

def create_feature_chart(features_history):
    if not features_history:
        return go.Figure()
    arr = np.array(features_history)
    fig = go.Figure()
    for i in range(4):
        fig.add_trace(go.Scatter(y=arr[:, i], mode="lines", name=f"Feature {i+1}"))
    fig.update_layout(title="Feature Evolution")
    return fig

def create_status_distribution_chart(history):
    if not history:
        return go.Figure()
    counts = {}
    for h in history:
        counts[h["status"]] = counts.get(h["status"], 0) + 1
    return go.Figure(go.Pie(labels=list(counts.keys()), values=list(counts.values()), hole=0.4))

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="main-header">
<h1>🌱 PVDF Plant Health Monitoring System</h1>
<p>AI-powered plant health analysis using PVDF sensors</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# UPDATE DATA
# ============================================
if st.session_state.monitoring_active:
    d = generate_simulated_data()
    st.session_state.history.append(d)
    st.session_state.features_history.append(d["features"])
    st.session_state.current_status = d["status"]
    st.session_state.health_score = d["health_score"]
    st.session_state.last_update = datetime.now()

# ============================================
# DASHBOARD
# ============================================
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    status = st.session_state.current_status
    card = "unknown-card"
    if status == "HEALTHY": card = "healthy-card"
    elif status == "PEST STRESS": card = "pest-card"
    elif status == "WATER STRESS": card = "water-card"

    st.markdown(f"""
    <div class="status-card {card}">
        <h2>{status}</h2>
        <h1>{st.session_state.health_score:.1f}%</h1>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.plotly_chart(
        create_health_gauge(st.session_state.health_score, status),
        use_container_width=True,
        key="health_gauge_chart"
    )

with col3:
    st.metric("Last Update", st.session_state.last_update.strftime("%H:%M:%S"))

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(
        create_feature_chart(list(st.session_state.features_history)),
        use_container_width=True,
        key="feature_evolution_chart"
    )

with col2:
    st.plotly_chart(
        create_status_distribution_chart(list(st.session_state.history)),
        use_container_width=True,
        key="status_distribution_chart"
    )

# ============================================
# DATA TABLE
# ============================================
st.markdown("### 📋 Recent Data")
if st.session_state.history:
    df = pd.DataFrame(list(st.session_state.history)[-20:])
    st.dataframe(df, use_container_width=True)

# ============================================
# AUTO REFRESH
# ============================================
if st.session_state.auto_refresh:
    time.sleep(st.session_state.refresh_interval)
    st.rerun()
