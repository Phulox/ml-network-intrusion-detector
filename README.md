# ML-Powered Network Intrusion Detector

This project is a mini **Network Intrusion Detection System (IDS)** that uses
a **Multilayer Perceptron (MLP)** to classify network flows as normal or attack,
with:

- Python + PyTorch for the ML model
- PostgreSQL to store alerts
- Flask dashboard to visualize alerts

## Architecture

1. **Preprocessing (Day 2)**

   - Clean UNW-NB15-style network flow dataset
   - Select features and save `train.csv`

2. **Model Training (Day 3)**

   - MLP with 2 hidden layers (128 → 64) + Sigmoid output
   - StandardScaler normalization (per-feature mean/std)
   - Saves:
     - `model/model.pt` (trained weights)
     - `model/scaler_mean.npy`
     - `model/scaler_scale.npy`

3. **Detector Engine (Day 4)**

   - `detector.py`:
     - loads model + scaler
     - uses `FeatureExtractor` with shared `FEATURES` list
     - scores flows with the MLP
     - writes high-score alerts to PostgreSQL (`alerts` table)

4. **Dashboard (Day 5)**

   - `dashboard.py`:
     - Flask app that queries `alerts` table
     - `templates/alerts.html` shows recent alerts in a web UI

5. **Evaluation (Day 6)**
   - `evaluate_model.py`:
     - computes accuracy, precision, recall, F1, confusion matrix
     - evaluates at multiple thresholds (0.5, 0.7, 0.8, 0.9)
     - saves results to `model/metrics.json`
