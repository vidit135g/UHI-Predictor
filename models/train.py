import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Define dataset path
file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "UHI_features.csv")

# Ensure dataset exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Dataset not found at {file_path}. Did feature engineering run successfully?")

# Load dataset
df = pd.read_csv(file_path)

# Print column names for debugging
print("Available Columns in Dataset:", df.columns.tolist())

# Ensure "UHI Index" column exists
target_col = "UHI Index"
if target_col not in df.columns:
    raise KeyError(f"❌ Column '{target_col}' not found in dataset. Check feature engineering step.")

# Drop non-numeric columns except the target
non_numeric_columns = df.select_dtypes(exclude=["number"]).columns.tolist()
df.drop(columns=non_numeric_columns, inplace=True, errors="ignore")

# Clean feature names (remove spaces, brackets, special characters)
df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]", "", regex=True)

# Rename target column to match new format
new_target_col = target_col.replace(" ", "")
df.rename(columns={target_col: new_target_col}, inplace=True)

# Define features and target
X = df.drop(columns=[new_target_col])  # Features
y = df[new_target_col]  # Target

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, enable_categorical=True)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"✅ Model trained successfully. MAE: {mae:.4f}, R²: {r2:.4f}")

# Ensure models directory exists
models_dir = os.path.dirname(__file__)
os.makedirs(models_dir, exist_ok=True)

# Save the trained model
model_path = os.path.join(models_dir, "UHI_xgboost.pkl")
joblib.dump(model, model_path)
print(f"✅ Model saved at {model_path}")