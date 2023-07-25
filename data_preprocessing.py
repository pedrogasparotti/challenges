    #!/usr/bin/env python
# coding: utf-8

# In[7]:

import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path):
    # Load data
    df = pd.read_csv(file_path, encoding='utf-16', delimiter='\t')
    return df

def handle_outliers(df):
    # Detect outliers using the Z-score method and remove them
    z_scores = stats.zscore(df['preco'])
    threshold = 3
    df = df[(z_scores < threshold)]

    return df

def preprocess_data(df):
    # Label encode the categorical features
    label_encoder_marca = LabelEncoder()
    label_encoder_modelo = LabelEncoder()
    label_encoder_versao = LabelEncoder()

    df['marca'] = label_encoder_marca.fit_transform(df['marca'])
    df['modelo'] = label_encoder_modelo.fit_transform(df['modelo'])
    df['versao'] = label_encoder_versao.fit_transform(df['versao'])

    # Standardize numerical features
    scaler = StandardScaler()
    X_numerical = scaler.fit_transform(df[['ano_de_fabricacao', 'ano_modelo', 'hodometro']])

    # Categorical features
    X_marca = df['marca'].values
    X_modelo = df['modelo'].values
    X_versao = df['versao'].values

    try: 
        return X_numerical, X_marca, X_modelo, X_versao, df['preco']

    except:
       return X_numerical, X_marca, X_modelo, X_versao  
