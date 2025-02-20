# data_processing.py
# ------------------------------
# PURPOSE: Provide reusable functions to load and clean various datasets.

import pandas as pd
import geopandas as gpd

def load_uhi_data(filepath):
    """
    Loads UHI data from a CSV file and returns a DataFrame.
    """
    df = pd.read_csv(filepath)
    return df

def load_weather_data(filepath):
    """
    Loads weather data from an Excel file (xlsx) and returns a DataFrame.
    """
    df = pd.read_excel(filepath)
    return df

def load_building_footprint(filepath):
    """
    Loads building footprint data from a KML or GeoPackage file and returns a GeoDataFrame.
    """
    gdf = gpd.read_file(filepath)
    return gdf
