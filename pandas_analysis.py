
import pandas as pd
import numpy as np
import sqlite3

# Connect to database
conn = sqlite3.connect("data/instamart.db")

# Load completed orders with customer and product details
df = pd.read_sql_query("""
    SELECT
        o.order_id,
        o.customer_id,
        c.city,
        p.category,
        o.quantity,
        o.discount_pct,
        o.order_amount,
        o.delivery_minutes
    FROM orders o
    JOIN customers c
        ON o.customer_id = c.customer_id
    JOIN products p
        ON o.product_id = p.product_id
    WHERE o.status = 'completed'
""", conn)

conn.close()

print("Total completed orders:", len(df))

# ==================================
# 1. CITY / CATEGORY REVENUE PIVOT
# ==================================

print("\n===== CITY / CATEGORY REVENUE =====")

pivot = pd.pivot_table(
    df,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    fill_value=0
)

print(pivot)

# ==================================
# 2. CITY-WISE STATISTICS
# ==================================

print("\n===== CITY-WISE STATISTICS =====")

city_stats = df.groupby("city").agg(
    total_revenue=("order_amount", "sum"),
    mean_order_amount=("order_amount", "mean"),
    order_count=("order_id", "count")
).round(2)

print(city_stats)

# ==================================
# 3. ORDER AMOUNT DISTRIBUTION
# ==================================

print("\n===== ORDER AMOUNT DISTRIBUTION =====")

amount = df["order_amount"]

print("Mean:", amount.mean())
print("Median:", amount.median())
print("Skewness:", amount.skew())

print("\nQuartiles:")
print(amount.quantile([0.25, 0.50, 0.75]))

# ==================================
# 4. OUTLIER DETECTION USING IQR
# ==================================

print("\n===== OUTLIER DETECTION =====")

Q1 = amount.quantile(0.25)
Q3 = amount.quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (amount < lower_limit) |
    (amount > upper_limit)
]

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)
print("Number of outliers:", len(outliers))

# ==================================
# 5. CORRELATION MATRIX
# ==================================

print("\n===== CORRELATION MATRIX =====")

numeric_columns = [
    "quantity",
    "discount_pct",
    "order_amount",
    "delivery_minutes"
]

correlation = df[numeric_columns].corr()

print(correlation.round(3))

print("\n===== PANDAS ANALYSIS COMPLETED =====")
