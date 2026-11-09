
import pandas as pd

# 1. Load cleaned datasets
customers = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")
products = pd.read_csv("data/clean/products.csv")

# 2. Convert dates and numerical columns
customers["signup_date"] = pd.to_datetime(customers["signup_date"])
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
orders["order_amount"] = pd.to_numeric(orders["order_amount"])
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

# 3. Define cutoff dates
cutoff_date = pd.Timestamp("2025-09-01")
snapshot_date = cutoff_date

# 4. Separate historical orders and September orders
historical_orders = orders[
    orders["order_ts"] < cutoff_date
].copy()

september_orders = orders[
    (orders["order_ts"] >= cutoff_date) &
    (orders["order_ts"] < pd.Timestamp("2025-10-01"))
].copy()

# 5. Customers who had at least one completed order
# before September 1
historical_completed = historical_orders[
    historical_orders["status"] == "completed"
].copy()

eligible_customers = historical_completed[
    "customer_id"
].unique()

# 6. Build historical customer features
# Only information available before September 1 is used

# Recency, frequency and monetary
rfm_features = historical_completed.groupby(
    "customer_id"
).agg(
    last_order_date=("order_ts", "max"),
    frequency=("order_id", "count"),
    monetary=("order_amount", "sum")
).reset_index()

rfm_features["recency"] = (
    cutoff_date - rfm_features["last_order_date"]
).dt.days

# Average discount and cancellation rate
historical_stats = historical_orders.groupby(
    "customer_id"
).agg(
    avg_discount=("discount_pct", "mean"),
    total_orders=("order_id", "count"),
    cancelled_orders=(
        "status",
        lambda x: (x == "cancelled").sum()
    )
).reset_index()

historical_stats["cancellation_rate"] = (
    historical_stats["cancelled_orders"] /
    historical_stats["total_orders"]
)

# Average delivery time for completed orders
delivery_features = historical_completed.groupby(
    "customer_id"
)["delivery_minutes"].mean().reset_index()

delivery_features.rename(
    columns={"delivery_minutes": "avg_delivery_time"},
    inplace=True
)

# Favourite category from historical completed orders
category_orders = historical_completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

category_counts = (
    category_orders.groupby(
        ["customer_id", "category"]
    )
    .size()
    .reset_index(name="category_count")
    .sort_values(
        ["customer_id", "category_count", "category"],
        ascending=[True, False, True]
    )
)

favourite_category = category_counts.drop_duplicates(
    "customer_id"
)[["customer_id", "category"]]

favourite_category.rename(
    columns={"category": "favourite_category"},
    inplace=True
)

# 7. Combine historical features
features = (
    rfm_features
    .merge(historical_stats, on="customer_id")
    .merge(delivery_features, on="customer_id")
    .merge(favourite_category, on="customer_id")
)

# 8. Add customer demographic information
features = features.merge(
    customers[
        ["customer_id", "city", "age", "gender", "signup_date"]
    ],
    on="customer_id",
    how="left"
)

# Customer tenure as of September 1
features["tenure_days"] = (
    cutoff_date - features["signup_date"]
).dt.days

# 9. Create target label using September activity
september_active = september_orders[
    september_orders["status"] == "completed"
]["customer_id"].unique()

features["lapsed"] = (
    ~features["customer_id"].isin(september_active)
).astype(int)

# 10. Remove unnecessary date columns
features.drop(
    columns=["last_order_date", "signup_date"],
    inplace=True
)

# 11. Validation
print("\n===== LAPSE DATASET RESULTS =====")

print("Eligible customers:", len(eligible_customers))
print("Feature dataset shape:", features.shape)

print("\nLapse distribution:")
print(features["lapsed"].value_counts())

print("\nLapse percentages:")
print(
    (features["lapsed"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nMissing values:")
print(features.isnull().sum())

print("\nFeature columns:")
print(features.columns.tolist())

# 12. Save output
features.to_csv(
    "outputs/lapse_dataset.csv",
    index=False
)

print("\nLapse dataset saved successfully!")
