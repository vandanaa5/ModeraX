import pandas as pd

# Load the dataset
df = pd.read_csv("../data/HateSpeechDatasetBalanced.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Display column names
print("\nColumn Names:")
print(df.columns)

# Dataset information
print("\nDataset Information:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())