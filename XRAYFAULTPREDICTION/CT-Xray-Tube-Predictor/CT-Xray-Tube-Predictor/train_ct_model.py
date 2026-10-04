"""
train_ct_model.py  (PHASE 3)

WHAT IT DOES
    Loads data/ct_dataset.csv, trains a Random Forest to predict the tube's
    health status, prints the scores, and saves the model.

WHY A RANDOM FOREST?
    It is many small decision trees voting together. It handles messy
    sensor data well and needs almost no tuning.

INPUT : data/ct_dataset.csv
OUTPUT: models/ct_failure_model.pkl  and  models/ct_confusion_matrix.png
RUN   : python train_ct_model.py
"""

import joblib
import matplotlib
matplotlib.use("Agg")                     # draw to a file, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split

SENSORS = ["tube_temperature", "oil_temperature", "voltage",
           "current", "fan_speed", "usage_hours"]
STATES = ["Healthy", "Warning", "Critical", "Failed"]   # order matters

data = pd.read_csv("data/ct_dataset.csv")
X = data[SENSORS]        # what the model sees
y = data["status"]       # what it must guess

# Keep 20% of rows hidden so we can test on data the model never saw.
# stratify=y keeps the same mix of states in both parts.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# "macro" = average the score across the 4 states equally
print(f"Accuracy : {accuracy_score(y_test, predictions):.3f}")
print(f"Precision: {precision_score(y_test, predictions, average='macro'):.3f}")
print(f"Recall   : {recall_score(y_test, predictions, average='macro'):.3f}")
print(f"F1 score : {f1_score(y_test, predictions, average='macro'):.3f}")

matrix = confusion_matrix(y_test, predictions, labels=STATES)
print("\nConfusion matrix (rows = truth, columns = guess):")
print(pd.DataFrame(matrix, index=STATES, columns=STATES))

ConfusionMatrixDisplay(matrix, display_labels=STATES).plot(cmap="Blues")
plt.title("CT tube model - confusion matrix")
plt.savefig("models/ct_confusion_matrix.png", dpi=120, bbox_inches="tight")

# Which sensors did the forest rely on most?
print("\nSensor importance:")
for name, score in sorted(zip(SENSORS, model.feature_importances_),
                          key=lambda pair: -pair[1]):
    print(f"  {name:18s} {score:.3f}")

joblib.dump(model, "models/ct_failure_model.pkl")
print("\nSaved models/ct_failure_model.pkl")
