import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from feature_config import FEATURES
from train_model import MLP


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/processed/train.csv")

X = df[FEATURES].values.astype("float32")
y = df["label"].values.astype("int64")  # 0 or 1

# Load scaler stats
scaler_mean = np.load("model/scaler_mean.npy")
scaler_scale = np.load("model/scaler_scale.npy")

def apply_scaler(x):
    return (x - scaler_mean) / scaler_scale

X = apply_scaler(X).astype("float32")

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)


# -----------------------------
# Load model
# -----------------------------
input_dim = len(FEATURES)
model = MLP(input_dim)
state_dict = torch.load("model/model.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()


# -----------------------------
# Evaluate at a given threshold
# -----------------------------
def eval_at_threshold(threshold: float):
    with torch.no_grad():
        probs = model(X_tensor).squeeze().numpy()  # probabilities in [0,1]

    y_pred = (probs >= threshold).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred)

    print(f"\n=== Threshold = {threshold:.2f} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion matrix [ [TN FP], [FN TP] ]:")
    print(cm)

    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }


if __name__ == "__main__":
    # Try multiple thresholds
    results = []
    for t in [0.5, 0.7, 0.8, 0.9]:
        res = eval_at_threshold(t)
        results.append(res)

    # Save metrics to a JSON file for the repo
    import json
    with open("model/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved metrics to model/metrics.json")
