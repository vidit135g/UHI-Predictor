import os
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging with a consistent format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize FastAPI app with metadata
app = FastAPI(
    title="UHI Predictor API",
    version="1.0",
    description="API for predicting the Urban Heat Island (UHI) Index."
)

# Define a Pydantic model for input validation
class PredictionInput(BaseModel):
    Longitude: float
    Latitude: float
    hour: int
    day: int
    month: int
    weekday: int

def load_model() -> joblib:
    """
    Load the trained XGBoost model from disk.
    Returns:
        The loaded model.
    Raises:
        FileNotFoundError: If the model file is not found.
    """
    # Construct an absolute path for the model file from the project root.
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_root, "models", "UHI_xgboost.pkl")
    logging.info(f"Looking for model file at: {model_path}")

    if not os.path.exists(model_path):
        logging.error(f"Model file not found at: {model_path}")
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = joblib.load(model_path)
    logging.info("Model loaded successfully.")
    return model

# Load the model at startup
try:
    model = load_model()
except Exception as e:
    logging.error("Error loading model: " + str(e))
    raise e

@app.post("/predict/", summary="Predict UHI Index", response_description="The predicted UHI Index.")
def predict(input_data: PredictionInput):
    """
    Predict the UHI Index based on the input features.

    **Input Fields:**
    - **Longitude**: Geographic longitude of the location.
    - **Latitude**: Geographic latitude of the location.
    - **hour**: Hour of the day (24-hour format).
    - **day**: Day of the month.
    - **month**: Month of the year.
    - **weekday**: Day of the week (Monday=0, Sunday=6).

    **Returns:**
        A JSON object containing the predicted UHI Index.
    """
    try:
        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        logging.info("Input DataFrame for prediction:")
        logging.info(input_df)
        # Generate prediction and convert to a native Python float
        prediction = float(model.predict(input_df)[0])
        return {"predicted_UHI_Index": prediction}
    except Exception as e:
        logging.error("Prediction error: " + str(e))
        raise HTTPException(status_code=500, detail="Prediction failed: " + str(e))

if __name__ == "__main__":
    import uvicorn
    logging.info("Starting UHI Predictor API")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
