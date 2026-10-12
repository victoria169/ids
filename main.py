import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

np.random.seed(67)
n_samples = 10000
n_features = 30

X = pd.DataFrame(np.random.rand(n_samples,n_features),columns = [f"feature_{i}" for i in range(n_features)])
#rn the features are useless but later can make them related to determining whether it is an attack or not
X['protocol'] = pd.Categorical(np.random.choice(['feat1','feat2','feat3'], size=n_samples))
#not a balanced dataset
y = np.random.choice([0,1], size = n_samples, p = [0.85, 0.15])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 67, stratify = y)

training_data = lgb.Dataset(X_train, label = y_train)
testing_data = lgb.Dataset(X_test, label = y_test, reference = training_data)

params = {
    'objective': 'binary',
    'metric': ['binary_logloss','auc'],
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 67,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging-freq': 1,
    'is_unbalance': True,
    'random_state': 67,
    'verbosity': -1
}

num_round = 1000
callbacks = [lgb.early_stopping(stopping_rounds=50),lgb.log_evaluation(period=50)]

model = lgb.train(
    params,
    training_data,
    num_boost_round = num_round,
    valid_sets = [training_data, testing_data],
    callbacks = callbacks
)

y_pred_probs = model.predict(X_test, num_iteration = model.best_iteration)
y_pred = (y_pred_probs >= 0.5).astype(int)

print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test,y_pred_probs):.4f}")

#currently this model works through complete guessing as the attack vs not attack was assigned randomly
#needs to be added on