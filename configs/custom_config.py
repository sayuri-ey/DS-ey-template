# custom modules

# external modules
import os
import time

def get_timezone():
    os.environ['TZ'] = 'America/Sao_Paulo'
    time.tzset()

def get_data_path():
    data_path = []
    return data_path

def get_key_columns()
    key_columns = []
    return key_columns

def get_target_column():
    target_column = []
    return target_column

def get_temporal_column():
    temporal_column = []
    return temporal_column

def get_date_columns():
    date_columns = []
    return date_columns

def get_feature_columns():
    feature_columns = []
    return feature_columns

def get_categorical_columns():
    cat_columns = []
    return cat_columns

def get_numerical_columns():
    num_columns = []
    return num_columns

def get_binary_columns():
    binary_columns = []
    return binary_columns

def get_str_to_num():
    str_to_num = []
    return str_to_num

def get_num_to_str():
    num_to_str = []
    return num_to_str

def get_columns_with_missing_values()
    cols_with_missing_values = []
    return cols_with_missing_values

def get_cols_to_drop():
    cols_to_drop = []
    return cols_to_drop

def store_data(data, file_name):

    file_path = f'./pipeline/data/pkl/{file_name}.pkl'

    with open(file_path, 'wb') as file:
        pickle.dump(data, file)