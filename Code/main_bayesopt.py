"""
    Based on LightGBM and Bayesian Optimization
    See "Feature selection_LightGBM_BayesianOptimization.ipynb"


"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import sklearn
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve
from utils import resumetable, reduce_mem_usage, fit_categorical_feature, data_preprocessing
import pickle

from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from bayes_opt import BayesianOptimization

df_train, df_test, Y_train = data_preprocessing("bayesopt", pre_loaded = True)
categorical =   ["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
                 "P_emaildomain", "R_emaildomain"] +  ["M" + str(i) for i in range(1, 10)] +\
                ['DeviceType', 'DeviceInfo', 'Weekday', 'Hour',
                 'P_emaildomain_pre', 'P_emaildomain_suffix', 'R_emaildomain_pre', 'R_emaildomain_suffix'] +\
                ["id_"+str(i) for i in range(12,38) if "id_"+str(i) in df_train.columns]
category_counts =  {'ProductCD': 6, 'card1': 17092, 'card2': 503, 'card3': 135, 'card4': 6, 'card5': 140,
                    'card6': 6, 'addr1': 443, 'addr2': 95, 'P_emaildomain': 62, 'R_emaildomain': 62, 'M1': 4,
                    'M2': 4, 'M3': 4, 'M4': 5, 'M5': 4, 'M6': 4, 'M7': 4, 'M8': 4, 'M9': 4, 'DeviceType': 4,
                    'DeviceInfo': 2801, 'Weekday': 8, 'Hour': 25, 'P_emaildomain_pre': 10, 'P_emaildomain_suffix': 10,
                    'R_emaildomain_pre': 10, 'R_emaildomain_suffix': 10, 'id_12': 4, 'id_13': 57, 'id_14': 30, 'id_15': 5,
                    'id_16': 4, 'id_17': 129, 'id_19': 570, 'id_20': 549, 'id_28': 4, 'id_29': 4, 'id_30': 88, 'id_31': 173,
                    'id_32': 8, 'id_33': 463, 'id_34': 6, 'id_35': 4, 'id_36': 4, 'id_37': 4, 'id_38': 4}
numerical =     ["TransactionAmt", "dist1"] + ["C" + str(i) for i in range(1, 15) if "C" + str(i) in df_train.columns] + \
                ["D" + str(i) for i in range(1, 16) if "D" + str(i) in df_train.columns] + ["PCA_V_"+str(i) for i in range(20)] + \
                ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_09', 'id_10', 'id_11']

scaler = StandardScaler()
scaler.fit(np.concatenate([df_train, df_test]))
scaled_x_train = scaler.transform(df_train)
scaled_x_test = scaler.transform(df_test)
X_train, X_cv, Y_train, Y_cv = train_test_split(scaled_x_train, Y_train, test_size=0.3, random_state = 42)
X_test = scaled_x_test

with open("../Data/X_train_v5.pkl", "wb") as handle:
    pickle.dump(X_train, handle)
with open("../Data/X_cv_v5.pkl", "wb") as handle:
    pickle.dump(X_cv, handle)
with open("../Data/X_test_v5.pkl", "wb") as handle:
    pickle.dump(X_test, handle)
with open("../Data/Y_train_v5.pkl", "wb") as handle:
    pickle.dump(Y_train, handle)
with open("../Data/Y_cv_v5.pkl", "wb") as handle:
    pickle.dump(Y_cv, handle)

lgb_train = lgb.Dataset(data=X_train.astype('float32'), label=Y_train.astype('float32'))
lgb_valid = lgb.Dataset(data=X_cv.astype('float32'), label=Y_cv.astype('float32'))

bounds = {
    'num_leaves': (31, 500),
    'min_data_in_leaf': (20, 200),
    'max_depth':(-1, 50),
    'bagging_fraction' : (0.1, 0.9),
    'feature_fraction' : (0.1, 0.9),
    'lambda_l1': (0, 2),
    'lambda_l2': (0, 2)
}

bo = BayesianOptimization(train_model, bounds, random_state=42)
bo.maximize(init_points=10, n_iter=15, acq='ucb', xi=0.0, alpha=1e-6)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': False,
    'boost_from_average': True,
    'num_threads': 4,
    
    'num_leaves': int(bo.max['params']['num_leaves']),
    'min_data_in_leaf': int(bo.max['params']['min_data_in_leaf']),
    'max_depth': int(bo.max['params']['max_depth']),
    'bagging_fraction' : bo.max['params']['bagging_fraction'],
    'feature_fraction' : bo.max['params']['feature_fraction'],
    'lambda_l1': bo.max['params']['lambda_l1'],
    'lambda_l2': bo.max['params']['lambda_l2']
}

lgb_model = lgb.train(params, lgb_train, valid_sets=lgb_valid, verbose_eval=1000)
Y_pred = lgb_model.predict(X_test.astype('float32'), num_iteration=lgb_model.best_iteration)
submission = pd.read_csv('../Data/sample_submission.csv', index_col='TransactionID')
submission['isFraud'] = Y_pred
submission.to_csv('../Data/Y_test_v5.csv')

# Showing features' importance graph
feature_importance_df = pd.concat([
    pd.Series(df_train.columns),
    pd.Series(lgb_model.feature_importance())], axis=1)
feature_importance_df.columns = ['featureName', 'importance']

# get top 100 features sorted by importance descending
temp = feature_importance_df.sort_values(by=['importance'], ascending=False).head(100)

plt.figure(figsize=(20, 20))
sns.barplot(x="importance", y="featureName", data=temp)
plt.xlabel("Importance", fontsize = 18)
plt.ylabel("FeatureName", fontsize = 18)
plt.show()


def train_model(num_leaves, min_data_in_leaf, max_depth, bagging_fraction, feature_fraction, lambda_l1, lambda_l2):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': False,
        'boost_from_average': True,
        'num_threads': 4,
        
        'num_leaves': int(num_leaves),
        'min_data_in_leaf': int(min_data_in_leaf),
        'max_depth': int(max_depth),
        'bagging_fraction' : bagging_fraction,
        'feature_fraction' : feature_fraction,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2
    }
    
    lgb_model = lgb.train(params, lgb_train, valid_sets=lgb_valid, verbose_eval=1000)
    
    y = lgb_model.predict(X_cv.astype('float32'), num_iteration=lgb_model.best_iteration)
    
    score = roc_auc_score(Y_cv.astype('float32'), y)
    return score
