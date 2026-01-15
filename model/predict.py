import pickle
import pandas as pd
import os
import sys

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pkl')

def load_model():
    """Load the trained LinearRegressionModel."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run train.py first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict(model, input_dict):
    """
    Predict price using the loaded model.
    input_dict must contain: name, company, year, kms_driven, fuel_type
    """
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)
    return prediction[0]

if __name__ == "__main__":
    # Example usage
    try:
        model = load_model()
        sample_input = {
            'name': 'Maruti Suzuki Swift', 
            'company': 'Maruti', 
            'year': 2019, 
            'kms_driven': 100, 
            'fuel_type': 'Petrol'
        }
        price = predict(model, sample_input)
        print(f"Predicted Price for {sample_input['name']}: {price}")
    except Exception as e:
        print(f"Error: {e}")
