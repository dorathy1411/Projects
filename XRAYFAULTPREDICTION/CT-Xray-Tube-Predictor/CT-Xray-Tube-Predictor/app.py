"""
app.py  (PHASE 5)

WHAT IT DOES
    A Streamlit web dashboard that shows the simulated CT tube live:
    six sensor values, health status, failure risk, and two history charts.

RUN:  streamlit run app.py

HOW IT WORKS
    Streamlit re-runs this whole file from top to bottom every time something
    changes. Values that must survive between re-runs (wear, history) live in
    st.session_state. While "Run live" is on, we wait one second and
    ask Streamlit to re-run, which gives one new reading per second.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from simulator import SENSORS, assess, load_model, next_reading

st.set_page_config(page_title="CT X-Ray Tube Monitor", layout="wide")

STATUS_COLOURS = {"Healthy": "#2e9e4f", "Warning": "#e0a800",
                  "Critical": "#e8590c", "Failed": "#c92a2a"}


@st.cache_resource          # load the model once, not every second
def get_model():
    return load_model()


def start_new_tube():
    """Reset everything back to a brand-new tube."""
    rng = np.random.default_rng()
    st.session_state.wear = 0.0
    st.session_state.usage_factor = rng.uniform(0.8, 1.2)
    st.session_state.history = []


if "history" not in st.session_state:
    start_new_tube()
model = get_model()

# ---------- sidebar controls ----------
st.sidebar.header("Controls")
running = st.sidebar.toggle("Run live (1 reading / second)", value=False)
speed = st.sidebar.slider("Wear speed", 1, 5, 2,
                          help="Higher = the tube wears out faster")
if st.sidebar.button("Reset to new tube"):
    start_new_tube()
    st.rerun()

# ---------- make one new reading ----------
# We only advance wear while running, so pausing freezes the picture.
if running and st.session_state.wear < 1.0:
    st.session_state.wear += 0.004 * speed
rng = np.random.default_rng()
reading = next_reading(min(st.session_state.wear, 1.0),
                       st.session_state.usage_factor, rng)
status, risk = assess(model, reading)

# Save to history only while running (or the very first reading)
if running or not st.session_state.history:
    st.session_state.history.append({**reading, "risk": risk})
history = pd.DataFrame(st.session_state.history)

# ---------- page ----------
st.title("CT X-Ray Tube Health Monitor")
st.caption("Simulated sensor data. Predicts machine health only, never patient data.")

colour = STATUS_COLOURS[status]
st.markdown(
    f"<div style='background:{colour};color:white;padding:14px;border-radius:8px;"
    f"font-size:28px;font-weight:600'>Status: {status} &nbsp;|&nbsp; "
    f"Failure risk: {risk}%</div>", unsafe_allow_html=True)
st.progress(min(int(risk), 100))

c1, c2, c3 = st.columns(3)
c1.metric("Tube temperature", f"{reading['tube_temperature']:.1f} °C")
c2.metric("Oil temperature", f"{reading['oil_temperature']:.1f} °C")
c3.metric("Voltage", f"{reading['voltage']:.1f} V")
c4, c5, c6 = st.columns(3)
c4.metric("Current", f"{reading['current']:.2f} A")
c5.metric("Fan speed", f"{reading['fan_speed']:.0f} rpm")
c6.metric("Usage hours", f"{reading['usage_hours']:.0f} h")

left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(history["tube_temperature"], label="Tube")
    ax.plot(history["oil_temperature"], label="Oil")
    ax.set_title("Temperatures (°C)"); ax.set_xlabel("Seconds"); ax.legend()
    st.pyplot(fig); plt.close(fig)
with right:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(history["risk"], color="#c92a2a")
    ax.set_ylim(0, 100)
    ax.set_title("Failure risk (%)"); ax.set_xlabel("Seconds")
    st.pyplot(fig); plt.close(fig)

if st.session_state.wear >= 1.0:
    st.error("The tube has reached end of life. Press 'Reset to new tube' to start over.")

# ---------- keep the live loop going ----------
if running and st.session_state.wear < 1.0:
    time.sleep(1)
    st.rerun()
