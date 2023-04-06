# custom modules
from configs.custom_config import get_timezone, get_target_column, get_feature_columns, get_categorical_columns, get_numerical_columns
# external modules
import joblib
import pandas as pd
from datetime import datetime

# Load model
model = joblib.load('model/model.pkl')

def predict(new_data):
    
    before = datetime.now()
    
    # Convert the new data to a pandas DataFrame
    df = pd.DataFrame([new_data])

    predictions = model.predict(df)
    
    after = datetime.now()
    
    message = f"new_data was labeled as {'fraude' if predictions == 1 else 'não-fraude'}. Inference took {(after-before).total_seconds()*1000:.0f} {'miliseconds' if (after-before).total_seconds()*1000 < 1000 else 'seconds'}"
    print(message)
    
    return predictions