"""Monitoring Dashboard & Alerts (Streamlit).

Run:  streamlit run app/dashboard.py
Modes: Simulation (progressive fault + maintenance feedback loop),
       Upload CSV, ESP32 live (via api/ingest_server.py).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.config import FAULTS, MODEL_PATH, SENSORS, UNITS, WINDOW_SIZE
from src.data_generator import generate_window, severity_to_state
from src.predict import PredictiveMaintenanceEngine
from src.train import train

st.set_page_config(page_title="Motor Health Monitor", page_icon="⚙️", layout="wide")


@st.cache_resource
def get_engine():
    if not MODEL_PATH.exists():
        with st.spinner("First run: training the model on simulated data..."):
            train(400, verbose=False)
    return PredictiveMaintenanceEngine()


engine = get_engine()
ss = st.session_state
ss.setdefault("history", [])
ss.setdefault("alerts", [])
ss.setdefault("actions", [])
ss.setdefault("severity", 0.0)
ss.setdefault("window", None)
ss.setdefault("result", None)
ss.setdefault("running", False)
ss.setdefault("rng", np.random.default_rng(11))
ss.setdefault("step", 0)

COLORS = {"HEALTHY": "🟢", "WARNING": "🟠", "CRITICAL": "🔴"}


def record(result, window, truth=None):
    ss.window, ss.result = window, result
    ss.step += 1
    prev = ss.history[-1]["health_state"] if ss.history else None
    ss.history.append({"step": ss.step, "risk_score": result["risk_score"],
                       "health_state": result["health_state"], "risk_level": result["risk_level"],
                       "true_state": truth})
    if result["alert"] and result["health_state"] != prev:
        ss.alerts.append({"time": time.strftime("%H:%M:%S"), "state": result["health_state"],
                          "risk": result["risk_level"], "fault": result["fault_type"],
                          "recommendation": result["recommendation"]})


# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Motor Health Monitor")
mode = st.sidebar.radio("Data source", ["Simulation", "Upload CSV", "ESP32 live (API)"])

if mode == "Simulation":
    fault = st.sidebar.selectbox("Fault type", FAULTS)
    rate = st.sidebar.slider("Degradation per step", 0.0, 0.15, 0.05, 0.01)
    interval = st.sidebar.slider("Step interval (s)", 0.2, 3.0, 1.0, 0.1)
    manual = st.sidebar.slider("Set fault severity", 0.0, 1.4, 0.0, 0.05)
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Apply severity"):
        ss.severity = manual
    if c2.button("Reset all"):
        for k in ("history", "alerts", "actions"):
            ss[k] = []
        ss.severity, ss.step, ss.result, ss.running = 0.0, 0, None, False
    c3, c4 = st.sidebar.columns(2)
    step_clicked = c3.button("Next reading")
    ss.running = c4.toggle("Auto-run", value=ss.running)
    if st.sidebar.button("🔧 Perform maintenance"):
        ss.actions.append({"time": time.strftime("%H:%M:%S"), "action": f"Maintenance on '{fault}'",
                           "severity_before": round(ss.severity, 2)})
        ss.severity = 0.0   # feedback loop: maintenance restores the motor
    if step_clicked or ss.running:
        ss.severity += rate
        win = generate_window(ss.severity, fault, ss.rng, start_ms=ss.step * 12800)
        record(engine.assess(win), win, severity_to_state(ss.severity))

elif mode == "Upload CSV":
    st.sidebar.caption(f"CSV columns: {', '.join(SENSORS)} (optional timestamp)")
    up = st.sidebar.file_uploader("Sensor CSV", type="csv")
    sample = generate_window(0.6, "bearing_wear", np.random.default_rng(1), window=WINDOW_SIZE * 3)
    st.sidebar.download_button("Sample CSV", sample.to_csv(index=False), "sample_motor_data.csv")
    if up is not None and st.sidebar.button("Analyse file"):
        df = pd.read_csv(up)
        ss.history = []
        for i in range(0, max(len(df) - WINDOW_SIZE + 1, 1), WINDOW_SIZE):
            win = df.iloc[i:i + WINDOW_SIZE]
            if len(win) >= 8:
                record(engine.assess(win), win)

else:
    url = st.sidebar.text_input("API URL", "http://localhost:5000")
    ss.running = st.sidebar.toggle("Auto-refresh", value=ss.running)
    interval = st.sidebar.slider("Refresh (s)", 1, 10, 3)
    if st.sidebar.button("Fetch latest") or ss.running:
        try:
            r = requests.get(f"{url}/latest", timeout=3)
            if r.ok:
                p = r.json()
                if not ss.history or ss.result != p["assessment"]:
                    record(p["assessment"], pd.DataFrame(p["samples"]))
            else:
                st.sidebar.warning("No data received yet.")
        except requests.RequestException as e:
            st.sidebar.error(f"API unreachable: {e}")

st.sidebar.divider()
m = engine.metrics
st.sidebar.caption(f"Model: {m['model']} · test accuracy {m['health_accuracy']:.1%} on {m['data']}")

# ---------------- Main page ----------------
st.title("AI-Based Predictive Maintenance – Industrial Motor")
res, win = ss.result, ss.window

if res is None:
    st.info("Use the sidebar to generate a reading (Simulation), upload a CSV or connect the ESP32.")
else:
    state = res["health_state"]
    banner = {"HEALTHY": st.success, "WARNING": st.warning, "CRITICAL": st.error}[state]
    banner(f"{COLORS[state]} Motor is **{state}** · failure risk **{res['risk_level']}** · "
           f"{res['recommendation']}")

    c = st.columns(4)
    c[0].metric("Health status", state)
    c[1].metric("Failure risk", res["risk_level"], f"score {res['risk_score']:.2f}", delta_color="off")
    c[2].metric("Likely fault", (res["fault_type"] or "none").replace("_", " "))
    c[3].metric("Priority", res["priority"])

    st.subheader("Real-time motor parameters")
    cols = st.columns(3)
    clean = win.reset_index(drop=True)
    for i, s in enumerate(SENSORS):
        if s in clean.columns:
            with cols[i % 3]:
                st.caption(f"{s.capitalize()} ({UNITS[s]}) · latest {clean[s].iloc[-1]:.2f}")
                st.line_chart(clean[s], height=140)

    left, right = st.columns(2)
    with left:
        st.subheader("Maintenance decision support")
        st.markdown(f"**Recommendation:** {res['recommendation']}")
        st.markdown(f"**Priority:** {res['priority']}")
        st.markdown(f"**Corrective action:** {res['corrective_action']}")
        st.caption("Top deviating indicators vs. healthy baseline")
        st.dataframe(pd.DataFrame(res["top_indicators"]), hide_index=True, width="stretch")
    with right:
        st.subheader("Failure-risk trend")
        h = pd.DataFrame(ss.history).set_index("step")
        st.line_chart(h["risk_score"], height=220)
        st.caption(f"Class probabilities: {res['probabilities']} · "
                   f"anomaly flag: {'YES' if res['anomaly'] else 'no'}")

    a, b = st.columns(2)
    with a:
        st.subheader("Alert log")
        st.dataframe(pd.DataFrame(ss.alerts) if ss.alerts else pd.DataFrame({"info": ["No alerts yet"]}),
                     hide_index=True, width="stretch")
    with b:
        st.subheader("Maintenance actions")
        st.dataframe(pd.DataFrame(ss.actions) if ss.actions else pd.DataFrame({"info": ["None performed"]}),
                     hide_index=True, width="stretch")

if ss.running:
    time.sleep(interval)
    st.rerun()
