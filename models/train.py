import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# ----------------- 1️⃣ Load & Preprocess Data -----------------
file_path = "data/processed/UHI_Weather_Building_Sentinel_LST_Featured_Cleaned.csv"
df = pd.read_csv(file_path)

print(f"✅ Dataset Loaded. Shape: {df.shape}")

# **Print Available Columns Before Processing**
print("📌 Available Columns in Dataset:", df.columns.tolist())

# **Sanitize column names (remove special characters)**
df.columns = df.columns.str.replace(r"[^\w\s]", "_", regex=True)
print("✅ Column Names Sanitized:", df.columns.tolist())

# **Drop datetime column** (since it's non-numeric)
if "datetime" in df.columns:
    df.drop(columns=["datetime"], inplace=True)
    print("✅ Removed `datetime` column.")

# ----------------- 2️⃣ Extract Temporal Features -----------------
if "hour" not in df.columns:
    df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
if "weekday" not in df.columns:
    df["weekday"] = pd.to_datetime(df["datetime"]).dt.weekday
if "month" not in df.columns:
    df["month"] = pd.to_datetime(df["datetime"]).dt.month

print("✅ Extracted Temporal Features: 'hour', 'weekday', 'month'")

# ----------------- 3️⃣ Create Missing Categorical Features -----------------
# ✅ **Convert wind direction into categories**
def categorize_wind_direction(degrees):
    if degrees >= 0 and degrees < 90:
        return "North-East"
    elif degrees >= 90 and degrees < 180:
        return "South-East"
    elif degrees >= 180 and degrees < 270:
        return "South-West"
    else:
        return "North-West"

wind_direction_col = [col for col in df.columns if "wind_direction" in col.lower()]
if wind_direction_col:
    df["wind_direction_category"] = df[wind_direction_col[0]].apply(categorize_wind_direction)
    print("✅ Wind Direction Categorized!")

# ✅ **Categorize time of day based on `hour` column**
df["hour_category"] = pd.cut(
    df["hour"],
    bins=[0, 6, 12, 18, 24],
    labels=["Night", "Morning", "Afternoon", "Evening"],
    include_lowest=True,
)

print("✅ Created `hour_category` for time-based analysis.")

# ----------------- 4️⃣ Encode Categorical Features -----------------
categorical_columns = ["hour_category", "wind_direction_category"]

# ✅ One-Hot Encoding
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_features = encoder.fit_transform(df[categorical_columns])

# Convert to DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# ✅ Save the encoder for prediction
os.makedirs("models", exist_ok=True)
joblib.dump(encoder, "models/encoder.pkl")
print("✅ Encoder saved at models/encoder.pkl")

# Drop original categorical columns & Concatenate encoded ones
df = df.drop(columns=categorical_columns).reset_index(drop=True)
df = pd.concat([df, encoded_df], axis=1)

print("✅ Categorical Features Encoded Successfully!")

# ----------------- 5️⃣ Feature Selection -----------------
# ✅ **Drop Unused Columns (Only If They Exist)**
unused_cols = ["Longitude", "Latitude", "Nearest_Building_Lon", "Nearest_Building_Lat"]
df = df.drop(columns=[col for col in unused_cols if col in df.columns], errors="ignore")

# **Ensure all features are numeric before training**
for col in df.columns:
    if df[col].dtype == "object":
        print(f"⚠️ Non-numeric column detected: {col} (Dropping)")
        df.drop(columns=[col], inplace=True)

# Define features (X) and target variable (y)
X = df.drop(columns=["uhi_index"])
y = df["uhi_index"]

# ✅ Save feature names for alignment in prediction
feature_names = list(X.columns)
joblib.dump(feature_names, "models/feature_names.pkl")
print("✅ Feature names saved.")

# ----------------- 6️⃣ Train-Test Split -----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Data Split: Train={X_train.shape}, Test={X_test.shape}")

# ----------------- 7️⃣ Model Training & Evaluation -----------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"📌 Model Performance: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    return mae, rmse, r2

# ✅ Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
print("✅ Random Forest Model Trained!")
evaluate_model(rf_model, X_test, y_test)

# ✅ Train XGBoost Model
xgb_model = XGBRegressor(objective="reg:squarederror", n_estimators=200, learning_rate=0.1, max_depth=9, random_state=42)
xgb_model.fit(X_train, y_train)
print("✅ XGBoost Model Trained!")
evaluate_model(xgb_model, X_test, y_test)

# ✅ Save trained models
joblib.dump(rf_model, "models/rf_model.pkl")
joblib.dump(xgb_model, "models/xgb_model.pkl")
print("✅ Models saved successfully!")

# ----------------- 8️⃣ Feature Importance Plot -----------------
def plot_feature_importance(model, feature_names, title):
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
    plt.title(title)
    plt.show()

plot_feature_importance(xgb_model, X.columns, "Feature Importance (XGBoost)")

print("✅ Model Training & Evaluation Complete!")