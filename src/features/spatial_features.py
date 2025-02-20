# spatial_features.py
# ------------------------------
# PURPOSE: Create or compute spatial features from your data

def extract_spatial_features(df):
    """
    Example: If there's a 'temperature' column, create a derived feature.
    Real logic might involve geospatial analysis, building density, NDVI, etc.
    """
    if 'temperature' in df.columns:
        df['temp_squared'] = df['temperature'] ** 2
    return df
