"""
generate_ct_data.py  (PHASE 2)

WHAT IT DOES
    Creates a fake (synthetic) CT X-ray tube dataset with 10,000 rows
    and saves it to data/ct_dataset.csv

WHY IT EXISTS
    Real hospital CT data is private, so we invent data that behaves the
    way a wearing-out tube plausibly would.

THE BIG IDEA
    Every row is one tube at one moment. We give each row a hidden number
    called "wear" from 0 (brand new) to 1 (dead). Wear is NOT given to the
    model. Instead, wear pushes the sensors in a believable direction:
        more wear -> hotter tube, hotter oil, lower voltage,
                     higher current, slower fan, more usage hours.
    Random noise is added so sensors are not perfect, just like real life.
    The health label comes from the hidden wear.

ASSUMPTIONS (documented, as the project requires)
    - Wear is spread evenly between 0 and 1.
    - Status cut-offs:  wear < 0.40 Healthy
                        0.40-0.70  Warning
                        0.70-0.90  Critical
                        >= 0.90    Failed
    - Sensor ranges are loosely based on the example in the project brief.
    - Usage hours vary +-20% between tubes (some tubes are used harder).
    - Noise is deliberate, so neighbouring states overlap and the model
      is not 100% perfect.

OUTPUT:  data/ct_dataset.csv  (columns: 6 sensors + status)
RUN:     python generate_ct_data.py
"""

import numpy as np
import pandas as pd

NUMBER_OF_ROWS = 10000
rng = np.random.default_rng(seed=42)   # fixed seed = same data every run


def wear_to_status(wear):
    """Turn the hidden wear number into a health label."""
    if wear < 0.40:
        return "Healthy"
    elif wear < 0.70:
        return "Warning"
    elif wear < 0.90:
        return "Critical"
    else:
        return "Failed"


def make_sensors(wear, rng):
    """
    Turn wear (a number or an array of numbers) into the 6 sensor values.
    simulator.py imports this function too, so live data uses the same rules.
    """
    n = np.size(wear)
    return {
        "tube_temperature": 70 + 30 * wear + rng.normal(0, 3.0, n),
        "oil_temperature":  60 + 30 * wear + rng.normal(0, 3.0, n),
        "voltage":          120 - 10 * wear + rng.normal(0, 1.5, n),
        "current":          8.0 + 2.5 * wear + rng.normal(0, 0.4, n),
        "fan_speed":        2200 - 700 * wear + rng.normal(0, 80, n),
        "usage_hours":      3200 * wear * rng.uniform(0.8, 1.2, n),
    }


if __name__ == "__main__":
    wear = rng.uniform(0, 1, NUMBER_OF_ROWS)
    data = pd.DataFrame(make_sensors(wear, rng)).round(2)
    data["usage_hours"] = data["usage_hours"].clip(lower=0).round(0)
    data["status"] = [wear_to_status(w) for w in wear]

    data.to_csv("data/ct_dataset.csv", index=False)
    print("Saved data/ct_dataset.csv")
    print(data["status"].value_counts())
    print(data.head())
