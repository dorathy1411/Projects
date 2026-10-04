"""
simulator.py  (PHASE 4)

WHAT IT DOES
    Pretends to be a live CT X-ray tube that slowly wears out. Every second
    it makes new sensor readings, hands them to the trained model, and
    prints the health status and failure risk.

WHY IT EXISTS
    We have no real machine to connect to. This lets us demo "live
    monitoring". app.py (the dashboard) imports the functions below.

HOW IT WORKS
    A hidden number "wear" starts at 0 (new tube) and creeps up each second.
    Sensor values come from the SAME rules used to build the training data
    (make_sensors in generate_ct_data.py), so the model recognises them.

FAILURE RISK %
    The model gives a probability for each state, e.g.
        Healthy 10%, Warning 70%, Critical 20%, Failed 0%
    We turn that into one risk number by giving each state a weight:
        Healthy 0, Warning 1/3, Critical 2/3, Failed 1
    and taking the weighted average. (Example above -> 37%.)
    This is a simple "how bad is it overall" score, not a calibrated
    real-world probability.

RUN:  python simulator.py
"""

import time

import joblib
import numpy as np
import pandas as pd

from generate_ct_data import make_sensors

SENSORS = ["tube_temperature", "oil_temperature", "voltage",
           "current", "fan_speed", "usage_hours"]
RISK_WEIGHTS = {"Healthy": 0.0, "Warning": 1 / 3, "Critical": 2 / 3, "Failed": 1.0}


def load_model(path="models/ct_failure_model.pkl"):
    """Load the trained Random Forest from disk."""
    return joblib.load(path)


def next_reading(wear, usage_factor, rng):
    """
    Make one set of sensor readings for a tube at the given wear (0 to 1).
    usage_factor (0.8-1.2) is picked once per run so usage_hours only
    goes up smoothly instead of jumping around.
    """
    values = make_sensors(wear, rng)
    reading = {name: float(values[name][0]) for name in SENSORS}
    reading["usage_hours"] = 3200 * wear * usage_factor
    return {name: round(value, 2) for name, value in reading.items()}


def assess(model, reading):
    """
    Give the sensor reading to the model.
    Returns (status, risk_percent).
    """
    row = pd.DataFrame([reading])[SENSORS]   # same column names as training
    probabilities = model.predict_proba(row)[0]
    chances = dict(zip(model.classes_, probabilities))

    status = max(chances, key=chances.get)                # most likely state
    risk = sum(RISK_WEIGHTS[state] * chance for state, chance in chances.items())
    return status, round(risk * 100, 1)


if __name__ == "__main__":
    model = load_model()
    rng = np.random.default_rng()
    usage_factor = rng.uniform(0.8, 1.2)
    wear = 0.0

    print("CT tube simulator - press Ctrl+C to stop\n")
    while wear < 1.0:
        reading = next_reading(wear, usage_factor, rng)
        status, risk = assess(model, reading)

        print(f"Current Status: {status}")
        print(f"Failure Risk:   {risk}%")
        print(f"Tube Temperature: {reading['tube_temperature']:.0f} C | "
              f"Oil: {reading['oil_temperature']:.0f} C | "
              f"Voltage: {reading['voltage']:.0f} V | "
              f"Current: {reading['current']:.1f} A | "
              f"Fan: {reading['fan_speed']:.0f} rpm | "
              f"Hours: {reading['usage_hours']:.0f}\n")

        wear += rng.uniform(0.01, 0.03)    # the tube degrades a little
        time.sleep(1)                      # one reading per second
