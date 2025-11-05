class FeatureExtractor:
    def __init__(self):
        self.feature_names = [
            "dur", "sbytes", "dbytes", "sttl", "dttl",
            "sload", "dload", "spkts", "dpkts",
            "swin", "dwin", "stcpb", "dtcpb", "smean", "dmean",
            "trans_depth", "res_bdy_len", "ct_srv_src", "ct_srv_dst",
            "is_ftp_login", "ct_flw_http_mthd"
        ]

    def transform(self, df):
        # tomorrow you’ll implement this
        pass