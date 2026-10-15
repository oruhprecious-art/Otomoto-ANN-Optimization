import pandas as pd
from pathlib import Path

# Locate the project directory
project_dir = Path(__file__).resolve().parent

# Locate the dataset
data_path = project_dir / "data" / "teleconnect.csv"

# Load the dataset
df = pd.read_csv(data_path)

print("=" * 50)
print("OTOMOTO DATASET EXPLORATION")
print("=" * 50)

print("\n1. Dataset dimensions:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n2. Column names:")
print(df.columns.tolist())

print("\n3. First five records:")
print(df.head())

print("\n4. Data types and missing values:")
print(df.info())

print("\n5. Missing values per column:")
print(df.isnull().sum())

print("\n6. Duplicate records:")
print(df.duplicated().sum())

print("\n7. Possible target columns:")
for column in df.columns:
    if any(word in column.lower()
           for word in ["churn", "target", "label"]):
        print(f"\nColumn: {column}")
        print(df[column].value_counts(dropna=False))