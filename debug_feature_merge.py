import pandas as pd

# Load UHI dataset
uhi_path = "data/raw/UHI_data.csv"  # Adjust if needed
uhi_df = pd.read_csv(uhi_path)

# Load weather dataset (where air temp, humidity, etc., should come from)
weather_path = "data/raw/NY_Mesonet_Weather.xlsx"  # Adjust if needed
weather_df = pd.read_csv(weather_path)

print("\n📌 First 5 Rows of UHI Dataset:")
print(uhi_df.head())

print("\n📌 First 5 Rows of Weather Dataset:")
print(weather_df.head())

print("\n📌 UHI Data - DateTime Column Format:")
print(uhi_df["datetime"].head())

print("\n📌 Weather Data - DateTime Column Format:")
print(weather_df["datetime"].head())

# Attempt merge
uhi_df["datetime"] = pd.to_datetime(uhi_df["datetime"], errors="coerce")
weather_df["datetime"] = pd.to_datetime(weather_df["datetime"], errors="coerce")

merged_df = uhi_df.merge(weather_df, on="datetime", how="left")

print("\n📌 After Merge - Missing Values:")
print(merged_df.isnull().sum())