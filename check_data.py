import pandas as pd

df = pd.read_csv("data/processed/UHI_features.csv")

print("📌 First 5 rows of dataset:")
print(df.head())

print("\n📌 Column Names:")
print(df.columns)

print("\n📌 Missing Values per Column:")
print(df.isnull().sum())