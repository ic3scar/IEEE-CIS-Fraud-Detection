import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
from sklearn import preprocessing
# import seaborn as sns
import pickle
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import LabelEncoder, StandardScaler, minmax_scale

def nn_data_preprocessing(pre_loaded = False):
    if pre_loaded:
        df_train = pd.read_pickle("../Data/Training_set_ver4.pkl")
        df_test = pd.read_pickle("../Data/Test_set_ver4.pkl")
        Y_train = pd.readpickle("../Data/Training_set_Y_ver4.pkl")
    else:
        df_identity_train = pd.read_csv("../Data/train_identity.csv")
        df_transaction_train = pd.read_csv("../Data/train_transaction.csv")
        df_identity_test = pd.read_csv("../Data/test_identity.csv")
        df_transaction_test = pd.read_csv("../Data/test_transaction.csv")

        df_train = pd.merge(df_transaction_train, df_identity_train, how = "left", on = "TransactionID")
        df_test = pd.merge(df_transaction_test, df_identity_test, how = "left", on = "TransactionID")
        Y_train = df_train["isFraud"]

        del df_identity_train, df_transaction_train, df_identity_test, df_transaction_test

        # Drop the columns that are missing more than 90% of the data or having a dominant value (low variance)
        cols_to_drop = drop_cols(df_train, df_test)
        df_train.drop(cols_to_drop, axis = 1, inplace = True)
        df_test.drop(cols_to_drop, axis = 1, inplace = True)
        # Make new categorical features "Weekday" and "Hour" based on "TransactionDT"
        df_train["Weekday"] = make_day_feature(df_train)
        df_train["Hour"]= make_hour_feature(df_train)
        df_test["Weekday"] = make_day_feature(df_test)
        df_test["Hour"] = make_hour_feature(df_test)
        # Adding new categorical feature regarding email domains
        df_train, df_test = add_emaildomain_feature(df_train, df_test)
        
        categorical = ["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
                       "P_emaildomain", "R_emaildomain"] +  ["M" + str(i) for i in range(1, 10)] +\
                      ['DeviceType', 'DeviceInfo', 'Weekday', 'Hour',
                       'P_emaildomain_pre', 'P_emaildomain_suffix', 'R_emaildomain_pre', 'R_emaildomain_suffix'] +\
                      ["id_"+str(i) for i in range(12,38) if "id_"+str(i) in df_train.columns]
        # Label encoding
        category_counts = fit_categorical_feature(df_train, df_test, categorical)
        # Using PCA on Vxxx features
        Vcols = []
        for i in range(1, 340):
            col = "V"+str(i)
            if col in df_train.columns:
                # print(col, df_train[col].min(), df_train[col].max(), df_train[col].nunique())
                Vcols.append(col)
        df = pd.concat([df_train, df_test], axis=0, join='outer')
        df = df.reset_index()
        for col in Vcols:
            df[col].fillna((df[col].min() - 2), inplace=True)
            df[col] = (minmax_scale(df[col], feature_range=(0,1)))  
        df = PCA_change(df, Vcols, prefix='PCA_V_', n_components=20) # n_components can be customized and should be hyper tuned
        # filling nan and applay StandardScaler for numerical features
        numerical = ["TransactionAmt", "dist1"] + ["C" + str(i) for i in range(1, 15)] + \
                    ["D" + str(i) for i in range(1, 16)] + ["PCA_V_"+str(i) for i in range(20)] + \
                    ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_09', 'id_10', 'id_11']
        numerical.remove("C3")
        numerical.remove("D7")
        for col in numerical:
            if df[col].isnull().sum()>0:
                df[col] = df[col].fillna(df_train[col].mean())
        for col in numerical:
            scaler = StandardScaler()
            if df[col].max() > 100 and df[col].min() >= 0:
                df[col] = np.log1p(df[col])
            scaler.fit(df[col].values.reshape(-1,1))
            df[col] = scaler.transform(df[col].values.reshape(-1,1))
        df = reduce_mem_usage(df)
        df_train = df.iloc[:590540, :]
        df_test = df.iloc[590540:, :]
        df_train.to_pickle("../Data/Training_set_ver4.pkl")
        df_test.to_pickle("../Data/Test_set_ver4.pkl")
        Y_train.to_pickle("../Data/Training_set_Y_ver4.pkl")
    return df_train, df_test, Y_train


def add_emaildomain_feature(df_train, df_test):
    emails =    {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 
                'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 
                'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 
                'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 
                'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 
                'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other',
                'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 
                'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 
                'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 
                'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 
                'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 
                'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other', 
                'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple', 
                'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 
                'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple',
                -999:"undefined"}
    us_emails = ['gmail', 'net', 'edu']
    # https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499#latest_df-579654
    for col in ['P_emaildomain', 'R_emaildomain']:
        df_train[col + '_pre'] = df_train[col].map(emails)
        df_test[col + '_pre'] = df_test[col].map(emails)

        df_train[col + '_suffix'] = df_train[col].map(lambda x: str(x).split('.')[-1])
        df_test[col + '_suffix'] = df_test[col].map(lambda x: str(x).split('.')[-1])

        df_train[col + '_suffix'] = df_train[col + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
        df_test[col + '_suffix'] = df_test[col + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    return df_train, df_test


def drop_cols(df_train, df_test):
    one_val_cols =      [col for col in df_train.columns if df_train[col].nunique()<=1] +\
                        [col for col in df_test.columns if df_test[col].nunique()<=1]
    missing_val_cols =  [col for col in df_train.columns if df_train[col].isnull().sum()/df_train.shape[0]>0.9] +\
                        [col for col in df_test.columns if df_test[col].isnull().sum()/df_test.shape[0]>0.9]
    same_val_cols =     [col for col in df_train.columns if df_train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9] +\
                        [col for col in df_test.columns if df_test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
    cols_to_drop = list(set(one_val_cols + missing_val_cols + same_val_cols))
    cols_to_drop.remove('isFraud')
    return cols_to_drop

def PCA_change(df, cols, n_components, prefix='PCA_', rand_seed=4):
    pca = PCA(n_components=n_components, random_state=rand_seed)

    principalComponents = pca.fit_transform(df[cols])

    principalDf = pd.DataFrame(principalComponents)

    df.drop(cols, axis=1, inplace=True)

    principalDf.rename(columns=lambda x: str(prefix)+str(x), inplace=True)

    df = pd.concat([df, principalDf], axis=1)
    
    return df

def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        # print(name)
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def make_day_feature(df, offset=0.58, tname='TransactionDT'):
    """
    Creates a day of the week feature, encoded as 0-6.
    """
    days = df[tname] / (3600 * 24)
    encoded_days = np.floor(days - 1 + offset) % 7
    return encoded_days

def make_hour_feature(df, tname='TransactionDT'):
    """
    Creates an hour of the day feature, encoded as 0-23.
    """
    hours = df[tname] / (3600)
    encoded_hours = np.floor(hours) % 24
    return encoded_hours

def fit_categorical_feature(df_train, df_test, categorical):
    # Label encoding
    category_counts = {}
    for col in categorical:
        df_train[col] = df_train[col].replace("nan", "other")
        df_train[col] = df_train[col].replace(np.nan, "other")
        df_test[col] = df_test[col].replace("nan", "other")
        df_test[col] = df_test[col].replace(np.nan, "other")
        
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_train[col].values) + list(df_test[col].values))
        df_train[col] = lbl.transform(list(df_train[col].values))
        df_test[col] = lbl.transform(list(df_test[col].values))
        category_counts[col] = len(list(lbl.classes_)) + 1    
    return category_counts

def get_input_features(df):
    X = {'numerical':np.array(df[numerical])}
    for col in categorical:
        X[col] = np.array(df[col])
    return X