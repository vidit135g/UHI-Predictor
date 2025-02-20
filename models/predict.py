import joblib
import pandas as pd
import os

# Define model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "UHI_xgboost.pkl")

def load_model():
    """Loads the trained XGBoost model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model file '{MODEL_PATH}' not found!")
    print("✅ Model loaded successfully.")
    return joblib.load(MODEL_PATH)

def preprocess_features(features: dict):
    """Processes input features and ensures exact order matching the model."""
    
    # Define expected order of features (MUST match training order)
    expected_features = [
        "Name", "Description", "location_id", "AirTempatSurfacedegC",
        "RelativeHumiditypercent", "AvgWindSpeedms", "WindDirectiondegrees",
        "SolarFluxWm2", "Latitude", "Longitude", "Altitude",
        "hour", "day", "month", "weekday"
    ]

    # Fill missing features with defaults
    default_values = {
        "Name": "Unknown",
        "Description": "None",
        "location_id": "0",
        "Latitude": 0.0,
        "Longitude": 0.0
    }
    
    for feature in expected_features:
        if feature not in features:
            features[feature] = default_values.get(feature, 0)

    # Convert to DataFrame & enforce correct column order
    df = pd.DataFrame([features])[expected_features]

    # Convert categorical features into numeric labels
    for cat_col in ["Name", "Description", "location_id"]:
        df[cat_col] = df[cat_col].astype("category").cat.codes

    return df

def predict(features: dict):
    """Prepares input features and makes a prediction."""
    model = load_model()

    # Preprocess input data
    df = preprocess_features(features)

    # Ensure model feature order is identical to df column order
    model_feature_order = model.get_booster().feature_names
    df = df[model_feature_order]  # Sort DataFrame columns to match model
    
    # Make prediction
    prediction = model.predict(df)[0]
    return prediction

if __name__ == "__main__":
    print("🔄 Running Prediction...")

    # Example input (Modify as needed)
    sample_input = {
        "Name": "Building A",
        "Description": "Commercial Area",
        "location_id": "123",
        "AirTempatSurfacedegC": 32.5,
        "RelativeHumiditypercent": 60,
        "AvgWindSpeedms": 2.5,
        "WindDirectiondegrees": 180,
        "SolarFluxWm2": 450,
        "Latitude": 26.8467,
        "Longitude": 80.9462,
        "Altitude": 120,
        "hour": 14,
        "day": 19,
        "month": 2,
        "weekday": 4
    }

    pred = predict(sample_input)
    print(f"🎯 Predicted UHI Index: {pred:.4f}")