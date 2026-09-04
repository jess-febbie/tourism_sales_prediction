import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH, index_col=0)

# Expected columns for the Tourism dataset
expected_columns = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "DurationOfPitch",
]

# Validate the dataset structure
missing = [c for c in expected_columns if c not in df.columns]

if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

# Dataset registration summary
print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nColumns:")
print(list(df.columns))

print("\nDataset information:")
df.info()

print("\nTarget variable distribution:")
print(df["ProdTaken"].value_counts())

print("\nTarget variable percentage distribution:")
print((df["ProdTaken"].value_counts(normalize=True) * 100).round(2))
