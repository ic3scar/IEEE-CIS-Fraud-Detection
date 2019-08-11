"""
    Neural Network (with embedding layer) for IEEE-CIS Fraud Detection
    Data pre-processing and model fitting
    For the description of the goal, see https://www.kaggle.com/c/ieee-fraud-detection/overview
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve
from utils import PCA_change, resumetable, reduce_mem_usage, make_day_feature, make_hour_feature, fit_categorical_feature, nn_data_preprocessing, get_input_features
from nn_model import construct_model
import pickle


df_train, df_test, Y_train = nn_data_preprocessing(pre_loaded = True)
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

# Showcase the data
print(resumetable(df_train))
print(resumetable(df_test))

X_train = get_input_features(df_train)
X_test  = get_input_features(df_test)
X_train, X_cv, Y_train, Y_cv = train_test_split(X_train, Y_train, test_size=0.3, random_state = 42)

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

model = construct_model(numerical, categorical, category_counts, pre_loaded = False)
filepath = "../Data/NN_EL_Model_ver2.h5"
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=30, min_lr=0.00001, verbose=1)
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callback_list = [checkpoint, reduce_lr]
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc']) # Can also use/add custom metrics

print(model.summary())

history = model.fit(X_train, Y_train, epochs = 100, batch_size = 1024, callbacks = callback_list)
preds = model.predict(X_train)
print("roc_auc_score on the training set: ", roc_auc_score(Y_train, preds))
print("F1 score on the training set: ", sklearn.metrics.f1_score(Y_train, preds>0.5))

# A simple graph of change of "loss" value
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

"""
    Tuning hyperparameters and features to see which ones to keep to maximize model's performance on the cross validation set
"""

# To predict based on the 
model2 = load_model("../Data/NN_EL_Model_ver2.h5")
predicts = model2.predict(X_test)
submission = pd.read_csv('../Data/sample_submission.csv', index_col='TransactionID')
submission['isFraud'] = predicts
submission.to_csv('../Data/Y_test_v4.csv')