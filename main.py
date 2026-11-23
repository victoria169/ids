import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

np.random.seed(67)
n_samples = 10000
n_features = 30

X = pd.DataFrame(np.random.rand(n_samples,n_features),columns = [f"feature_{i}" for i in range(n_features)])

#tcp, udp, and icmp are traffie features that can indicate risk but are not a 100% guarantee of an attack
#ill see if there are other features later
#these are abbrev. so you can search up the terms to better understand them
X['protocol'] = pd.Categorical(np.random.choice(['tcp','udp','icmp'], size=n_samples))
#more balanced dataset
#udp is least likely to be attack and icmp is most likely to be attack
protocol_effect = X["protocol"].map({"tcp": 0.6, "udp": -0.3, "icmp": 1.1}).astype(float)

#right now these features are js blank variables but can later assign them things like connection duration, byte counts, failed login counts, etc
#note that there are multiple different "features" that can point to an attack
#the features above this code are protocols, but other features should be added to increase deduction ability
logit = (
    -3.2  # baseline log odds when features and ref protocol effect is 0; starts low to keep overall attack prevalence 15% 
    + 2.8 * X["feature_0"] #positive, increases attack probability
    + 2.4 * X["feature_1"]
    - 1.5 * X["feature_2"] #negative, decreases attack probability
    + 2.2 * (X["feature_0"] * X["feature_1"])  #non-linear interaction
    + protocol_effect
)

#converts each sample's risk (logit) into attack probability
attack_prob = 1.0 / (1.0 + np.exp(-logit))
#flips a weighted coin to assign final label for attack
y = np.random.binomial(n=1, p=attack_prob)

#x is the given input features and y is the answer (either 0 or 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 67, stratify = y)

training_data = lgb.Dataset(X_train, label = y_train)
testing_data = lgb.Dataset(X_test, label = y_test, reference = training_data)

params = {
    'objective': 'binary',
    'metric': ['binary_logloss','auc'],
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 67,
    'num_iterations': 50
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging-freq': 1,
    'is_unbalance': True,
    'random_state': 67,
    'verbosity': -1,
    'min_data_in_leaf': 20,
    'min_sum_hessian_in_leaf': 0.001,
    'min_gain_to_split': 0.0
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

#how to read results (kinda)
#0=normal traffic and 1=attack
#precision is how often the model is correct when predicting 0 or 1
    #e.g. if the precision for 0 is 0.78 that means 78% of the time the model correctly identified normal traffice
        #the other 22% of the time it wrongly flagged attacks as normal traffic
#recall is out of all 0 or 1 what fraction did the model catch
#f1 is harmoic mean of precision and recall; how good the score is depends on how balanced your dataset is
#micro avg is teh unweighted simple average across classes
#weight average is the weighted average across classes (based on support)
    #support for both shows how many samples were used as tests (the rest are for training)
#support is the actual number of attacks that were either 0 or 1
#roc-auc [0.50,1.00] where 0.50 is guessing and 1.00 is perfect classification
