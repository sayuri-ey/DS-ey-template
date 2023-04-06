# custom modules

# external modules
import os
import time

def get_timezone():
    os.environ['TZ'] = 'America/Sao_Paulo'
    time.tzset()

def get_data_path():
    data_path = ''
    return data_path

def get_target_column():
    target_column = ''
    return target_column

def get_feature_columns():
    feature_columns = ''
    return feature_columns

def get_categorical_columns():
    cat_columns = ''
    return cat_columns

def get_numerical_columns():
    num_columns = ''
    return num_columns