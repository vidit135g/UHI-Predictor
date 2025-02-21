import pickle
import pandas as pd
import joblib
import numpy as np

# 🔹 Load trained models
rf_model = joblib.load("models/rf_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")

# 🔹 Load feature names used in training
with open("models/feature_names.pkl", "rb") as f:
    train_feature_names = pickle.load(f)

# ✅ Define User Input Dictionary (Modify based on actual user input)
user_inputs = {
    'land_surface_temp': 38.5,
    'band1': 1700,
    'band2': 1100,
    'band3': 1500,
    'band4': 1800,
    'air_temp_at_surface_degc': 30.0,
    'relative_humidity_percent': 60.0,
    'avg_wind_speed_m_s': 3.2,
    'wind_direction_degrees': 180.5,  # Example wind direction
    'solar_flux_w_m^2': 500,
    'nearest_building_lon': -73.93,
    'nearest_building_lat': 40.71,
    'building_distance_m': 50,
    'building_density_50m': 5,
    'building_density_100m': 10,
    'building_density_200m': 20,
    'hour': 15,
    'weekday': 5,
    'month': 7,
    'hour_category': "Afternoon",  # Categorical value
    'is_weekend': 0,
    'temp_gradient': 5.0,
    'humidity_temp_interaction': 1.5,
    'wind_chill': 25.0,
    'ndvi': 0.3,
    'ndbi': 0.6,
    'surface_albedo': 0.25
}

# ✅ Convert Input Dict to DataFrame
df_input = pd.DataFrame([user_inputs])

# 🔹 **Ensure Consistent Feature Encoding** (Match Training Encoding)
# ✅ One-Hot Encode Categorical Columns (Ensure all categories are present)
categorical_mappings = {
    "hour_category": ["Morning", "Afternoon", "Evening", "Night"],
    "wind_direction_degrees": [118.5, 121.6, 124.5, 124.7, 124.9, 125.3, 125.7, 126.1, 126.4, 126.5, 127.8, 128.3, 128.5, 130.2, 130.9, 132.1, 134.0, 138.5, 148.5, 154.0, 154.1, 154.2, 154.3, 154.4, 154.5, 157.7, 158.5, 159.0, 160.9, 164.0, 164.1, 167.3, 168.5, 169.0, 170.5, 172.0, 172.5, 174.0, 174.5, 175.5, 176.5, 178.5, 179.0, 180.4, 180.5, 181.8, 182.5, 183.2, 184.6, 186.0]
}

for col, categories in categorical_mappings.items():
    for category in categories:
        encoded_col_name = f"{col}_{category}"
        df_input[encoded_col_name] = (df_input[col] == category).astype(int) if col in df_input.columns else 0

df_input.drop(columns=["hour_category", "wind_direction_degrees"], errors="ignore", inplace=True)

# ✅ Ensure Missing Columns are Filled with 0 (Prevents KeyError)
for col in train_feature_names:
    if col not in df_input.columns:
        df_input[col] = 0

# ✅ Align Input Data with Training Data Columns
df_input = df_input[train_feature_names]

# ✅ Make Predictions
rf_pred = rf_model.predict(df_input)[0]
xgb_pred = xgb_model.predict(df_input)[0]

# ✅ Display Results
print(f"📌 Predicted UHI Index (Random Forest): {rf_pred:.6f}")
print(f"📌 Predicted UHI Index (XGBoost): {xgb_pred:.6f}")