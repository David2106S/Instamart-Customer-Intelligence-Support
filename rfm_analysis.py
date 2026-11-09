
import sqlite3
import pandas as pd

# ==========================================
# PART B - RFM CUSTOMER ANALYSIS
# ==========================================

DB_PATH = "data/instamart.db"

SNAPSHOT_DATE = pd.Timestamp("2025-10-01")

# Connect to SQLite database
conn = sqlite3.connect(DB_PATH)

# Load completed orders
orders = pd.read_sql_query(
    """
    SELECT
        customer_id,
        order_ts,
        order_amount
    FROM orders
    WHERE status = 'completed'
    """,
    conn
)

conn.close()

# Convert order timestamp into datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# ==========================================
# RFM CALCULATION
# ==========================================

rfm = orders.groupby("customer_id").agg(
    last_order_date=("order_ts", "max"),
    frequency=("order_ts", "count"),
    monetary=("order_amount", "sum")
).reset_index()

# Recency = days since last completed order
rfm["recency"] = (
    SNAPSHOT_DATE - rfm["last_order_date"]
).dt.days

# Reorder columns
rfm = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary"
    ]
]

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n===== RFM CUSTOMER ANALYSIS =====")

print("\nFirst 10 customers:")
print(rfm.head(10))

print("\nRFM dataset shape:")
print(rfm.shape)

print("\nRFM statistics:")
print(rfm[["recency", "frequency", "monetary"]].describe())

print("\nMissing values:")
print(rfm.isnull().sum())

# ==========================================
# VALIDATION
# ==========================================

print("\n===== RFM VALIDATION =====")

print("Unique RFM customers:", len(rfm))

assert len(rfm) == 1175, (
    f"Expected 1175 customers, got {len(rfm)}"
)

assert not rfm.isnull().any().any(), (
    "RFM dataset contains missing values"
)

assert (rfm["recency"] >= 0).all(), (
    "Negative recency detected"
)

assert (rfm["frequency"] > 0).all(), (
    "Invalid frequency detected"
)

# Save output
rfm.to_csv("outputs/rfm_analysis.csv", index=False)

print("\nRFM analysis saved successfully!")
print("ALL RFM CHECKPOINTS PASSED")
