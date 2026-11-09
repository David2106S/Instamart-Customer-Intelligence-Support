
import pandas as pd

# Load customer dataset
df = pd.read_csv("data/raw/customers.csv")

print("Missing values:")
print(df.isnull().sum())

print("\nMissing percentage:")
print((df.isnull().mean() * 100).round(2))

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nUnique city values:")
print(df["city"].value_counts())

print("\nUnique gender values:")
print(df["gender"].value_counts(dropna=False))

print("\nAge statistics:")
print(df["age"].describe())



print("\nInvalid ages (outside 10-100):")
print(((df["age"] < 10) | (df["age"] > 100)).sum())

print("\nDistinct raw city spellings:")
print(df["city"].nunique())

print("\nDistinct non-blank gender spellings:")
print(df["gender"].dropna().nunique())

print("\nBlank gender records:")
print(df["gender"].isna().sum())
