import pandas as pd 

df = pd.read_csv("data/raw/UNSW_NB15_training-set.csv")

print(df.head())
print(df.info())
print(df.columns)
print(df['label'].value_counts())