#!/usr/bin/env python
# coding: utf-8

# custom modules
from configs.custom_config import get_timezone, get_data_path, get_target_column, get_feature_columns, get_categorical_columns, get_numerical_columns
# external modules
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from datetime import datetime
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample


# set environment and define functions

# set timezone
get_timezone()

def get_dataset():
    dataset = pd.read_csv(get_data_path())
    return dataset

def split_data(df, test_size=0.50, validation_size=0.25, random_state=42):
    
    target = get_target_column()
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], test_size=test_size, random_state=random_state, stratify=df[target])
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=random_state, stratify=y_train)

    # Combine the train, validation, and test sets into a single dataframe
    train = pd.concat([X_train, y_train], axis=1)
    val = pd.concat([X_val, y_val], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    return train, val, test

def handle_missing_values(df):

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

def extract_date_info(df_input, date_cols):
    df = df_input
    list_date_feats = []
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        df[f"{col}_hour"] = df[col].dt.hour
        df.drop(col, axis=1, inplace=True)  
        list_date_feats.append([f"{col}_day", f"{col}_month", f"{col}_dayofweek", f"{col}_hour"])
    return df, list_date_feats

def extract_comission(df, value_col):
    df_out = df
    df_out[f"{value_col}_comission"] = df_out[value_col]*0.1
    return df_out

def transform_categorical(df, cols_to_encode):
    le = LabelEncoder()
    for col in cols_to_encode:
        df[col] = le.fit_transform(df[col])
    return df

def scale_numerical(df, cols_to_scale):
    scaler = MinMaxScaler()
    scaled_cols = scaler.fit_transform(df[cols_to_scale])
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaled_cols
    return df_scaled

def get_correlation_matrix(X_train, X_test, X_valid, threshold=0.6):
    corr_matrix = X_train.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    uncorrelated_X_train = X_train.drop(columns=drop_cols)
    uncorrelated_X_test = X_test.drop(columns=drop_cols)
    uncorrelated_X_valid = X_valid.drop(columns=drop_cols)
    print("Dropped columns:", drop_cols)
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.show()
    return uncorrelated_X_train, uncorrelated_X_test, uncorrelated_X_valid

def downsample(df, target_col, features, ratio=0.05):

    # Separate the majority and minority classes
    majority = df[df[target_col] == 0]
    minority = df[df[target_col] == 1]

    # Undersample the majority class
    n_minority = len(minority)
    n_majority = int(n_minority * ratio)
    majority_downsampled = resample(majority, replace=False, n_samples=n_majority, random_state=42)

    # Combine the minority and undersampled majority classes
    df_downsampled = pd.concat([minority, majority_downsampled])
    y_test = df_downsampled[target_col]
    df_downsampled = df_downsampled[features]

    return df_downsampled, y_test

def oversample_SMOTE(df_train, target_col, features):
    X_train = df_train[features]
    y_train = df_train[target_col]
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def oversample_ADASYN(df_train, target_col, features):

    X_train = df_train[features]
    y_train = df_train[target_col]
    adasyn = ADASYN(random_state=42)
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def split_features_target(df, target_col, features):
    cols = df.columns.tolist()
    pattern = re.compile(r"_train$|_test$|_valid$")
    new_cols = [re.sub(pattern, "", col) for col in cols]
    df.rename(columns=dict(zip(cols, new_cols)), inplace=True)
    X = df[features]
    Y = df[target_col]
    return X, Y

def main():
    
    print(f'Started Preprocess step at {datetime.now().strftime("%H:%M:%S")}')
    main_df = get_dataset()
    target_col = get_target_column()
    features = get_feature_columns()

    # Split dataset into training, validation, and test sets
    print(f'{datetime.now().strftime("%H:%M:%S")}: Splitting dataset')
    df_train, df_valid, df_test = split_data(main_df, test_size=0.50, validation_size=0.25, random_state=42)

    print(f'{datetime.now().strftime("%H:%M:%S")}: Starting feature processing')
    steps = {'train': df_train, 
             'test': df_valid, 
             'valid': df_test}
    for step, step_df in steps.items():
        print(f'{datetime.now().strftime("%H:%M:%S")}: Pre processing {step} set')
        missing_df = handle_missing_values(step_df)
        date_df = extract_date_info(missing_df, ['fecha'])
        encoded_df = transform_categorical(date_df, get_categorical_columns())
        scaled_df = scale_numerical(encoded_df,get_numerical_columns())
        df_final = scaled_df
        steps[step] = df_final.rename(columns={col: col + "_" + step for col in df_final.columns})    

    print(f'Train sets with {len(df_train)} rows')
    print(f'Test sets with {len(df_test)} rows')
    print(f'Valid sets with {len(df_valid)} rows')

    print(f'{datetime.now().strftime("%H:%M:%S")}: Split features and target sets and downsample train set')
    X_train, y_train = oversample_ADASYN(df_train, target_col, features)
    # X_train, y_train = split_features_target(df_train, target_col, features)
    X_test, y_test = split_features_target(df_test, target_col, features)
    X_valid, y_valid = split_features_target(df_valid, target_col, features)
    
    print(f'{datetime.now().strftime("%H:%M:%S")}: Checking feature correlation')
    X_train, X_test, X_valid = get_correlation_matrix(X_train, X_test, X_valid)
    
    # Save sets to csv
    print(f'{datetime.now().strftime("%H:%M:%S")}: Saving sets to csv')
    print(f'Train sets with {len(X_train)} rows')
    print(f'Test sets with {len(X_test)} rows')
    print(f'Valid sets with {len(X_valid)} rows')

    X_train.to_csv(f'data/train.csv', index=False, header=True, encoding='utf-8')
    X_test.to_csv(f'data/test.csv', index=False, header=True, encoding='utf-8')
    X_valid.to_csv(f'data/valid.csv', index=False, header=True, encoding='utf-8')
    y_train.to_csv(f'data/y_train.csv', index=False, header=True, encoding='utf-8')
    y_test.to_csv(f'data/y_test.csv', index=False, header=True, encoding='utf-8')
    y_valid.to_csv(f'data/y_valid.csv', index=False, header=True, encoding='utf-8')

    print(f'Finished Preprocess step at {datetime.now().strftime("%H:%M:%S")}')


if __name__ == "__main__":

    main()

