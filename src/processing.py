import pandas as pd 
from feature_config import FEATURES

df = pd.read_csv("data/raw/UNSW_NB15_training-set.csv")

X = df[FEATURES]
Y = df["label"]

X = X.astype("float32")
X = X.fillna(X.median())

out = pd.concat([X,Y], axis=1)
out.to_csv("data/processed/train.csv", index=False)
print("Success")