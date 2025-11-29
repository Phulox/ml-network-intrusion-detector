import numpy as np
from feature_config import FEATURES

class FeatureExtractor:
    def __init__(self):
        self.feature_names = FEATURES

    def transform(self, df):
        values = [df[f] for f in self.feature_names]
        return np.array(values, dtype="float32")
