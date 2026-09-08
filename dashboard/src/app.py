"""
Minimal operator dashboard, polling twin-core (:8001) and ml-models (:8002).
Run with: streamlit run src/app.py
"""

import time

import requests
import streamlit as st

TWIN_CORE_URL = "http://localhost:8001/telemetry/latest"
ML_MODELS_URL = "http://localhost:8002/health/latest"

st.set_page_config(page_title="Engine Digital Twin", layout="wide")
st.title("🛩️ Aero Piston Engine — Digital Twin (SIH26054)")

placeholder = st.empty()

while True:
    try:
        frame = requests.get(TWIN_CORE_URL, timeout=3).json()
        health = requests.get(ML_MODELS_URL, timeout=3).json()
    except requests.exceptions.RequestException:
        placeholder.warning("Waiting for twin-core / ml-models APIs to come online…")
        time.sleep(2)
        continue

    with placeholder.container():
        st.caption(f"Unit {frame['unit_id']} · cycle {frame['cycle']} · phase: {frame['phase']}")

        cols = st.columns(4)
        s = frame["sensors"]
        cols[0].metric("RPM", f"{s['rpm']:.0f}")
        cols[1].metric("CHT (°C)", f"{s['cht_c']:.1f}", delta=None)
        cols[2].metric("EGT (°C)", f"{s['egt_c']:.1f}")
        cols[3].metric("Oil Press (psi)", f"{s['oil_press_psi']:.1f}")

        st.divider()

        if health["anomaly"]["is_anomalous"]:
            st.error(f"⚠️ Anomaly detected (score {health['anomaly']['score']:.2f})")
        else:
            st.success("✅ Nominal operation")

        st.metric("Estimated RUL (cycles)", f"{health['rul_estimate_cycles']:.0f}")
        st.write("**Fault probabilities:**", health["fault_probabilities"])
        st.info(health["advisory"])

    time.sleep(1)
