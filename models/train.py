import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# ----------------- 1️⃣ Load & Preprocess Data -----------------
file_path = "data/processed/UHI_Weather_Building_Sentinel_LST_Featured_Cleaned.csv"
df = pd.read_csv(file_path)

print(f"✅ Dataset Loaded. Shape: {df.shape}")

# **Sanitize column names**
df.columns = df.columns.str.replace(r"[^\w\s]", "_", regex=True)

# **Drop datetime column**
df = df.drop(columns=["datetime"], errors="ignore")

# ----------------- 2️⃣ Extract Temporal Features -----------------
if "hour" not in df.columns:
    df["hour"] = pd.to_datetime(df["datetime"], errors="coerce").dt.hour
if "weekday" not in df.columns:
    df["weekday"] = pd.to_datetime(df["datetime"], errors="coerce").dt.weekday
if "month" not in df.columns:
    df["month"] = pd.to_datetime(df["datetime"], errors="coerce").dt.month

# ----------------- 3️⃣ Encode Categorical Features -----------------
categorical_columns = ["hour_category", "wind_direction_category"]

# ✅ One-Hot Encoding (Fix Applied)
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
if set(categorical_columns).issubset(df.columns):
    encoded_features = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

    # Save encoder for predictions
    os.makedirs("models", exist_ok=True)
    joblib.dump(encoder, "models/encoder.pkl")

    # Drop original categorical columns & concatenate encoded ones
    df = df.drop(columns=categorical_columns, errors="ignore").reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)

# ----------------- 4️⃣ Ensure All Features are Numeric -----------------
for col in df.columns:
    if df[col].dtype == "object":
        print(f"⚠️ Non-numeric column detected: {col} (Dropping)")
        df.drop(columns=[col], inplace=True)

# Convert all features to float to avoid XGBoost error
df = df.astype(float)

# ----------------- 5️⃣ Define Features and Target -----------------
X = df.drop(columns=["uhi_index"], errors="ignore")
y = df["uhi_index"]

# ✅ Save feature names for alignment in prediction
feature_names = list(X.columns)
joblib.dump(feature_names, "models/feature_names.pkl")

# ----------------- 6️⃣ Train-Test Split -----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
evaluate_model(rf_model, X_test, y_test)

# ✅ Train XGBoost Model (Fix Applied)
xgb_model = XGBRegressor(objective="reg:squarederror", n_estimators=200, learning_rate=0.1, max_depth=9, random_state=42)
xgb_model.fit(X_train.astype(float), y_train)  # 🔥 Explicitly Convert to Float
evaluate_model(xgb_model, X_test.astype(float), y_test)  # 🔥 Convert Test Data to Float

# ✅ Save trained models
joblib.dump(rf_model, "models/rf_model.pkl")
joblib.dump(xgb_model, "models/xgb_model.pkl")

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