# train.py
# ------------------------------
# PURPOSE: Script-based approach to train the model outside notebooks.

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def train_model():
    df = pd.read_csv("data/processed/UHI_features.csv")
    X = df.drop(columns=["UHI_Index","Longitude", "Latitude"])
    y = df["UHI_Index"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")

    joblib.dump(model, "models/UHI_xgboost.pkl")
    print("Model saved to models/UHI_xgboost.pkl")

if __name__ == "__main__":
    train_model()
