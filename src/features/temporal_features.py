# temporal_features.py
# ------------------------------
# PURPOSE: Extract time-based features (e.g., hour, day-of-week)

import pandas as pd

def extract_temporal_features(df):
    """
    If 'timestamp' column exists, convert to datetime and extract hour.
    """
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
    return df
