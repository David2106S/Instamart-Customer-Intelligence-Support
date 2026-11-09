
import pandas as pd

# Load product dataset
df = pd.read_csv("data/raw/products.csv")

print("===== PRODUCTS DATA AUDIT =====")

print("\nDataset shape:")
print(df.shape)

print("\nFirst five records:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nUnique product categories:")
print(df["category"].value_counts())

print("\nNumerical statistics:")
print(df.describe())

print("\n===== AUDIT COMPLETED =====")
