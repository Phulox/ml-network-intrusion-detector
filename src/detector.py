import numpy as np
import pandas as pd
from feature_extractor import FeatureExtractor
from train_model import MLP
import torch
import psycopg2
import torch.nn as nn
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", 0.8))


# -----------------------------
# Setup Postgres connection
# -----------------------------
def get_pg_conn():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )

conn = get_pg_conn()
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMP,
    src_ip VARCHAR(50),
    dst_ip VARCHAR(50),
    sport INT,
    dport INT,
    proto VARCHAR(10),
    score REAL
)
""")
conn.commit()


# -----------------------------
# Load scaler + model
# -----------------------------
scaler_mean = np.load("model/scaler_mean.npy")
scaler_scale = np.load("model/scaler_scale.npy")

def apply_scaler(x):
    return (x - scaler_mean) / scaler_scale


fe = FeatureExtractor()
input_dim = len(fe.feature_names)

model = MLP(input_dim)
state_dict = torch.load("model/model.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()    # IMPORTANT: evaluation mode


# -----------------------------
# Process a single flow
# -----------------------------
def process_flow(flow: dict):
    # 1) Extract features
    features = fe.transform(flow)
    features = apply_scaler(features).astype("float32")

    # 2) Run model prediction
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        score = model(x).item()

    # 3) Log & write alert if needed
    ts = datetime.utcnow()

    if score >= ALERT_THRESHOLD:
        print(f"[ALERT] score={score:.3f}")

        cur.execute("""
            INSERT INTO alerts (ts, src_ip, dst_ip, sport, dport, proto, score)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            ts,
            flow.get("src_ip", ""),
            flow.get("dst_ip", ""),
            int(flow.get("sport", 0)),
            int(flow.get("dport", 0)),
            flow.get("proto", ""),
            score,
        ))

        conn.commit()
    else:
        print(f"[OK]    score={score:.3f}")



# -----------------------------
# Simulate flows from CSV
# -----------------------------
def simulate_from_csv(path="data/processed/train.csv", n=20):
    df = pd.read_csv(path)
    subset = df.head(n)

    for _, row in subset.iterrows():
        flow = {f: float(row[f]) for f in fe.feature_names}

        flow["src_ip"] = "10.0.0.5"
        flow["dst_ip"] = "192.168.1.10"
        flow["sport"] = 50000
        flow["dport"] = 22
        flow["proto"] = "tcp"

        process_flow(flow)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    try:
        simulate_from_csv()
    finally:
        cur.close()
        conn.close()