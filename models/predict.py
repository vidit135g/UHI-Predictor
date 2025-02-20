import joblib
import pandas as pd
import sys
import os

def load_model():
    # Get the directory of this file (should be the project root if predict.py is here)
    project_root = os.path.dirname(os.path.abspath(__file__))
    # Build the path to the model file (which is in the "models" folder at the project root)
    model_path = os.path.join(project_root, "UHI_xgboost.pkl")
    print("Looking for model file at:", model_path)
    
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found!")
        sys.exit(1)
    
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    return model

def predict(features: dict):
    model = load_model()
    # Convert the input dictionary into a DataFrame
    df = pd.DataFrame([features])
    print("Input DataFrame for prediction:")
    print(df)
    # Make a prediction
    prediction = model.predict(df)[0]
    return prediction

if __name__ == "__main__":
    print("Running predict.py")
    # Sample input: adjust these keys to match the features used during training.
    sample_input = {
        "Longitude": 80.9462,  # Lucknow longitude
        "Latitude": 26.8467,   # Lucknow latitude
        "hour": 14,            # local hour (24-hour format)
        "day": 19,             # sample day (adjust as appropriate)
        "month": 2,            # sample month (adjust as appropriate)
        "weekday": 4           # weekday as integer (Monday=0, Sunday=6; adjust accordingly)
    }
    pred = predict(sample_input)
    print("Predicted UHI Index:", pred)
