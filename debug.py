import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Load Data
FILE_PATH = "data/processed/UHI_features.csv"
df = pd.read_csv(FILE_PATH)

# ✅ Rename Columns
column_mapping = {
    "Air Temp at Surface [degC]": "AirTempSurface",
    "Relative Humidity [percent]": "RelativeHumidity",
    "Avg Wind Speed [m/s]": "AvgWindSpeed",
    "Wind Direction [degrees]": "WindDirection",
    "Solar Flux [W/m^2]": "SolarFlux",
    "Altitude": "Altitude",
    "hour": "hour",
    "day": "day",
    "month": "month",
    "weekday": "weekday",
    "UHI Index": "UHI_Index"
}
df.rename(columns=column_mapping, inplace=True)

# ✅ Check for Missing Values
print("🔍 Checking for missing values...")
print(df.isnull().sum())

# ✅ Check Data Types
print("\n🔍 Checking data types...")
print(df.dtypes)

# ✅ Check Unique Values (Detect Constant Columns)
print("\n🔍 Checking unique values per column...")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# ✅ Ensure correct numeric types
df = df.astype({col: "float" for col in df.columns if col not in ["datetime", "Name", "Description", "location_id"]})

# ✅ Select Features and Target
X = df[[col for col in column_mapping.values() if col != "UHI_Index"]]
y = df["UHI_Index"]

# ✅ Check for Constant Values
if y.nunique() == 1:
    raise ValueError("❌ Target variable (UHI_Index) has only one unique value. Model can't learn!")

# ✅ Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train a Baseline Model (Before XGBoost)
print("\n🔍 Running a simple baseline test...")
y_baseline_pred = np.mean(y_train)  # Predicting mean value for all
baseline_r2 = r2_score(y_test, [y_baseline_pred] * len(y_test))
print(f"Baseline R² Score: {baseline_r2:.4f}")

# ✅ Save Debug File
df.to_csv("data/debug_UHI_features.csv", index=False)
print("\n✅ Debug file saved as 'data/debug_UHI_features.csv'")