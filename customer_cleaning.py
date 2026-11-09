
import pandas as pd
import numpy as np

# 1. Load original customer data
df = pd.read_csv("data/raw/customers.csv")

print("Original shape:", df.shape)

# 2. Standardise city names
df["city"] = df["city"].str.strip().str.title()

# 3. Standardise gender
df["gender"] = df["gender"].str.strip().str.lower()

gender_mapping = {
    "m": "male",
    "male": "male",
    "f": "female",
    "female": "female"
}

df["gender"] = df["gender"].map(gender_mapping).fillna("unknown")

# 4. Handle invalid ages
df.loc[(df["age"] < 10) | (df["age"] > 100), "age"] = np.nan

print("\nMissing ages after invalid-age handling:")
print(df["age"].isna().sum())

# 5. Impute missing ages using city median
df["age"] = df["age"].fillna(
    df.groupby("city")["age"].transform("median")
)

# 6. Convert signup_date to datetime
df["signup_date"] = pd.to_datetime(df["signup_date"])

# 7. Create output directory if needed
import os
os.makedirs("data/clean", exist_ok=True)

# 8. Save cleaned customer data
df.to_csv("data/clean/customers.csv", index=False)

# 9. Display validation results
print("\n--- CUSTOMER CLEANING RESULTS ---")

print("\nShape:", df.shape)

print("\nCity counts:")
print(df["city"].value_counts())

print("\nGender counts:")
print(df["gender"].value_counts())

print("\nAge statistics:")
print(df["age"].describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSignup date data type:")
print(df["signup_date"].dtype)

print("\nCleaned customer data saved successfully!")
