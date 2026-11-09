
import pandas as pd
import numpy as np
import os

# =====================================
# 1. LOAD RAW DATA
# =====================================

orders = pd.read_csv("data/raw/orders.csv")
products = pd.read_csv("data/raw/products.csv")

print("Original orders:", len(orders))

# =====================================
# 2. CONVERT DATA TYPES
# =====================================

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# =====================================
# 3. REMOVE EXACT DUPLICATES
# =====================================

orders = orders.drop_duplicates()

print("After removing exact duplicates:", len(orders))

# =====================================
# 4. KEEP LATEST RECORD PER ORDER ID
# =====================================

orders = orders.sort_values("order_ts")

orders = orders.drop_duplicates(
    subset="order_id",
    keep="last"
)

print("After removing duplicate order IDs:", len(orders))

# =====================================
# 5. REPAIR MISSING ORDER AMOUNTS
# =====================================

orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left"
)

missing_amounts = orders["order_amount"].isna().sum()

calculated_amount = (
    orders["quantity"]
    * orders["price_inr"]
    * (1 - orders["discount_pct"] / 100)
)

orders["order_amount"] = orders["order_amount"].fillna(
    calculated_amount
)

print("Missing amounts repaired:", missing_amounts)

# =====================================
# 6. FIX DELIVERY TIMES
# =====================================

# Negative delivery times are invalid
orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = np.nan

# Calculate category-wise median using valid completed orders
category_medians = (
    orders.loc[
        (orders["status"] == "completed")
        & orders["delivery_minutes"].notna()
    ]
    .groupby("category")["delivery_minutes"]
    .median()
)

# Impute missing delivery times for completed orders only
completed_missing = (
    (orders["status"] == "completed")
    & orders["delivery_minutes"].isna()
)

orders.loc[completed_missing, "delivery_minutes"] = (
    orders.loc[completed_missing, "category"]
    .map(category_medians)
)

# Cancelled orders must have missing delivery times
orders.loc[
    orders["status"] == "cancelled",
    "delivery_minutes"
] = np.nan

# =====================================
# 7. REMOVE TEMPORARY PRODUCT COLUMNS
# =====================================

orders = orders.drop(columns=["price_inr", "category"])

# =====================================
# 8. SAVE CLEANED DATA
# =====================================

os.makedirs("data/clean", exist_ok=True)

orders.to_csv(
    "data/clean/orders.csv",
    index=False
)

# =====================================
# 9. VALIDATION
# =====================================

print("\n===== ORDERS CLEANING RESULTS =====")

print("Total unique orders:", len(orders))

print("\nOrder status distribution:")
print(orders["status"].value_counts())

print("\nMissing values:")
print(orders.isnull().sum())

print(
    "\nMissing completed delivery times:",
    orders.loc[
        orders["status"] == "completed",
        "delivery_minutes"
    ].isna().sum()
)

print(
    "\nCancelled orders with delivery times:",
    orders.loc[
        orders["status"] == "cancelled",
        "delivery_minutes"
    ].notna().sum()
)

print(
    "\nCompleted order revenue:",
    round(
        orders.loc[
            orders["status"] == "completed",
            "order_amount"
        ].sum(),
        2
    )
)

print("\nCleaned orders saved successfully!")
