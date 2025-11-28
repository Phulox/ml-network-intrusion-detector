import pandas as pd 

df = pd.read_csv("data/raw/UNSW_NB15_training-set.csv")

FEATURES = [
    "dur",
    "spkts", "dpkts",
    "sbytes", "dbytes",
    "rate",
    "sttl", "dttl",
    "sload", "dload",
    "sloss", "dloss",
    "sinpkt", "dinpkt",
    "sjit", "djit",
    "swin", "dwin",
    "stcpb", "dtcpb",
    "tcprtt", "synack", "ackdat",
    "smean", "dmean",
    "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "ct_srv_dst",
    "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "is_sm_ips_ports",
]

X = df[FEATURES]
Y = df["label"]

X = X.astype("float32")
X = X.fillna(X.median())

out = pd.concat([X,Y], axis=1)
out.to_csv("data/processed/train.csv", index=False)
print("Success")