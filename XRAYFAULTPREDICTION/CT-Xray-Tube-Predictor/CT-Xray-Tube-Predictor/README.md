# CT X-Ray Tube Failure Prediction System

A beginner-friendly machine learning project that predicts the **health state**
(Healthy / Warning / Critical / Failed) and a **failure risk %** for a CT
scanner X-ray tube from sensor readings. It predicts machine health only,
never patient data or images.

## Quick start
```bash
pip install -r requirements.txt
python generate_ct_data.py      # creates data/ct_dataset.csv
python train_ct_model.py        # trains and saves models/ct_failure_model.pkl
streamlit run app.py            # opens the live dashboard
```
Optional: `python simulator.py` shows the same live simulation in the terminal.
Phase 1 notebook: open `notebooks/predictive_maintenance_analysis.ipynb` in VS Code or Jupyter.

## What is in each file
| File | Purpose |
|---|---|
| `notebooks/predictive_maintenance_analysis.ipynb` | Phase 1: learn on the AI4I 2020 dataset (Logistic Regression vs Random Forest) |
| `generate_ct_data.py` | Phase 2: builds 10,000 rows of synthetic CT sensor data |
| `train_ct_model.py` | Phase 3: trains and evaluates the Random Forest |
| `simulator.py` | Phase 4: live wear-out simulation + prediction |
| `app.py` | Phase 5: Streamlit dashboard |

## Synthetic data assumptions
- Each row has a hidden "wear" value (0 = new, 1 = dead), spread evenly.
- More wear means hotter tube and oil, lower voltage, higher current, slower fan, more usage hours.
- Status cut-offs on wear: <0.40 Healthy, 0.40-0.70 Warning, 0.70-0.90 Critical, >=0.90 Failed.
- Random noise is added so neighbouring states overlap, as in real life.
- This is invented data. The model learns our assumptions, not real tube physics.

## Failure risk %
Weighted average of the model's state probabilities (Healthy 0, Warning 1/3, Critical 2/3, Failed 1).
It is a severity score, not a calibrated real-world probability.

## Results
- CT model (test set): accuracy 0.906, macro precision 0.885, recall 0.887, F1 0.886.
- AI4I (failure vs OK): Random Forest accuracy 0.980 but recall only 0.471; Logistic Regression recall 0.824 but precision 0.142. Failures are rare (3.4%), so accuracy alone misleads.
