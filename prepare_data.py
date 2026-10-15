import pandas as pd
from pathlib import Path

# Locate the project directory
project_dir = Path(__file__).resolve().parent

# Load the original dataset
data_path = project_dir / "data" / "teleconnect.csv"
df = pd.read_csv(data_path)

print("=" * 60)
print("OTOMOTO DATA PREPARATION")
print("=" * 60)

# 1. Inspect TotalCharges
print("\n1. TotalCharges before conversion:")
print("Data type:", df["TotalCharges"].dtype)

# Convert TotalCharges to numeric.
# Invalid or whitespace-only values become NaN.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce"
)

print("\n2. TotalCharges after conversion:")
print("Data type:", df["TotalCharges"].dtype)
print("Missing TotalCharges:", df["TotalCharges"].isna().sum())

# 2. Examine the target variable
print("\n3. Churn distribution:")
print(df["Churn"].value_counts())

print("\n4. Churn percentages:")
print(
    (df["Churn"].value_counts(normalize=True) * 100)
    .round(2)
)

# 3. Remove the customer identifier
df = df.drop(columns=["customerID"])

# 4. Remove rows with missing target values
df = df.dropna(subset=["Churn"])

# 5. Handle missing TotalCharges values
# Median imputation is applied if conversion creates missing values.
missing_total = df["TotalCharges"].isna().sum()

if missing_total > 0:
    median_total = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_total)

print("\n5. Final dataset dimensions:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n6. Remaining missing values:")
print(df.isnull().sum())

print("\n7. Final data types:")
print(df.dtypes)

# 6. Save the cleaned dataset
output_path = project_dir / "data" / "teleconnect_clean.csv"
df.to_csv(output_path, index=False)

print("\nCleaned dataset saved to:")
print(output_path)