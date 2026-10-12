import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

rng = np.random.default_rng()
data = rng.uniform(size=(500, 10))  # 500 entities, each contains 10 features
label = rng.integers(low=0, high=2, size=(500, ))  # binary target
train_data = lgb.Dataset(data, label=label)
