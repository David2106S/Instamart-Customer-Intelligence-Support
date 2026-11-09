
import pandas as pd

# Load raw orders
df = pd.read_csv("data/raw/orders.csv")

print("===== ORDERS DATA AUDIT =====")

# 1. Dataset dimensions
print("\nDataset shape:")
print(df.shape)

# 2. First five records
print("\nFirst five records:")
print(df.head())

# 3. Column data types
print("\nData types:")
print(df.dtypes)

# 4. Missing values
print("\nMissing values:")
print(df.isnull().sum())

# 5. Missing percentage
print("\nMissing percentage:")
print((df.isnull().mean() * 100).round(2))

# 6. Exact duplicate rows
print("\nExact duplicate rows:")
print(df.duplicated().sum())

# 7. Duplicate order IDs
print("\nDuplicate order IDs:")
print(df["order_id"].duplicated().sum())

# 8. Order status distribution
print("\nOrder status distribution:")
print(df["status"].value_counts(dropna=False))

# 9. Numerical statistics
print("\nNumerical statistics:")
print(df.describe())

# 10. Unique categories
print("\nUnique order statuses:")
print(df["status"].unique())

print("\n===== AUDIT COMPLETED =====")
