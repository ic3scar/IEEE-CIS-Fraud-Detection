import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
from sklearn import preprocessing
import math
import tensorflow as tf
from keras import layers, regularizers
from keras.layers import Dense, Activation, Input, Concatenate, Embedding, Flatten, Dropout, BatchNormalization, SpatialDropout1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.initializers import glorot_uniform
import keras.backend as K
from keras.models import Model, load_model, Sequential
from keras import optimizers

import sklearn
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve

def construct_model(numerical, categorical, category_counts, pre_loaded = False):
    if pre_loaded:
        model = load_model("../Data/NN_EL_Model_ver2.h5")
    else:
        model = FC_NeuralNetwork_with_EmbeddingLayer(numerical, categorical, category_counts)
    return model

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def FC_NeuralNetwork_with_EmbeddingLayer(numerical, categorical, category_counts):
    categorical_inputs = []
    for col in categorical:
        categorical_inputs.append(Input(shape=[1], name = col))
        
    categorical_embeddings = []
    for i, col in enumerate(categorical):
        categorical_embeddings.append(
            Embedding(category_counts[col], int(np.log1p(category_counts[col])+1), name = col+"_emb")(categorical_inputs[i]))
    
    categorical_logits = Concatenate(name = "categorical_conc")(
        [Flatten()(SpatialDropout1D(.1)(col_emb)) for col_emb in categorical_embeddings])
    
    numerical_inputs = Input(shape=[df_train_v4[numerical].shape[1]], name = "numerical")
    numerical_logits = Dropout(.1)(numerical_inputs)
    
    x = Concatenate()([categorical_logits, numerical_logits])
    
    x = Dense(200, activation = 'relu')(x)
    x = Dropout(.2)(x)
    x = Dense(150, activation = 'relu')(x)
    x = Dropout(.2)(x)
    output = Dense(1, activation = 'sigmoid')(x)
    
    model = Model(inputs = categorical_inputs + [numerical_inputs], outputs = output)
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
    return model