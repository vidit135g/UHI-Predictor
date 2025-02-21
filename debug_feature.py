import pandas as pd

# Load the original dataset before feature engineering
raw_data_path = "data/raw/UHI_data.csv"  # Change this if needed
df = pd.read_csv(raw_data_path)

print("\n📌 First 5 Rows of Raw Dataset:")
print(df.head())

print("\n📌 Column Names Before Feature Engineering:")
print(df.columns)

print("\n📌 Missing Values Before Processing:")
print(df.isnull().sum())

# Check if feature extraction is working
processed_data_path = "data/processed/UHI_features.csv"
processed_df = pd.read_csv(processed_data_path)

print("\n📌 First 5 Rows of Processed Dataset:")
print(processed_df.head())

print("\n📌 Missing Values in Processed Dataset:")
print(processed_df.isnull().sum())