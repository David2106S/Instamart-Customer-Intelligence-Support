
import sqlite3
from pathlib import Path

import pandas as pd

# ==========================================
# PART A - DATA FOUNDATION VALIDATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

CUSTOMERS_FILE = BASE_DIR / "data" / "clean" / "customers.csv"
PRODUCTS_FILE = BASE_DIR / "data" / "clean" / "products.csv"
ORDERS_FILE = BASE_DIR / "data" / "clean" / "orders.csv"
DATABASE_FILE = BASE_DIR / "data" / "instamart.db"
SQL_FILE = BASE_DIR / "sql" / "queries.sql"

EXPECTED_REVENUE = 2998668.80

passed = 0
failed = 0


def check(description, condition):
    global passed, failed

    if condition:
        print(f"[PASS] {description}")
        passed += 1
    else:
        print(f"[FAIL] {description}")
        failed += 1


print("\n======================================")
print(" PART A - VALIDATION STARTED")
print("======================================\n")

# ==========================================
# 1. CHECK REQUIRED FILES
# ==========================================

print("1. FILE VALIDATION")

check("Clean customers CSV exists", CUSTOMERS_FILE.exists())
check("Clean products CSV exists", PRODUCTS_FILE.exists())
check("Clean orders CSV exists", ORDERS_FILE.exists())
check("SQLite database exists", DATABASE_FILE.exists())
check("SQL queries file exists", SQL_FILE.exists())

# Stop if essential files are missing
if not all([
    CUSTOMERS_FILE.exists(),
    PRODUCTS_FILE.exists(),
    ORDERS_FILE.exists(),
    DATABASE_FILE.exists(),
    SQL_FILE.exists()
]):
    print("\nValidation stopped: required files are missing.")
    raise SystemExit(1)

# ==========================================
# 2. LOAD CLEANED DATA
# ==========================================

customers = pd.read_csv(CUSTOMERS_FILE)
products = pd.read_csv(PRODUCTS_FILE)
orders = pd.read_csv(ORDERS_FILE)

print("\n2. CLEANED DATA VALIDATION")

check("Customers contain 1200 rows", len(customers) == 1200)
check("Products contain 40 rows", len(products) == 40)
check("Orders contain 12309 rows", len(orders) == 12309)

check(
    "Customer IDs are unique",
    customers["customer_id"].is_unique
)

check(
    "Product IDs are unique",
    products["product_id"].is_unique
)

check(
    "Order IDs are unique",
    orders["order_id"].is_unique
)

check(
    "No duplicate customer rows",
    not customers.duplicated().any()
)

check(
    "No duplicate product rows",
    not products.duplicated().any()
)

# ==========================================
# 3. CUSTOMER DATA VALIDATION
# ==========================================

print("\n3. CUSTOMER DATA VALIDATION")

check(
    "No missing customer values",
    not customers.isnull().any().any()
)

check(
    "Customer ages are between 18 and 60",
    customers["age"].between(18, 60).all()
)

check(
    "Customer genders are standardised",
    set(customers["gender"].unique()).issubset(
        {"male", "female", "unknown"}
    )
)

expected_cities = {
    "Chennai",
    "Mumbai",
    "Bengaluru",
    "Delhi",
    "Hyderabad",
    "Pune"
}

check(
    "Customer cities are standardised",
    set(customers["city"].unique()) == expected_cities
)

# ==========================================
# 4. ORDER DATA VALIDATION
# ==========================================

print("\n4. ORDER DATA VALIDATION")

check(
    "No missing order values except delivery time",
    not orders.drop(columns=["delivery_minutes"]).isnull().any().any()
)

check(
    "No negative delivery times",
    (orders["delivery_minutes"].dropna() >= 0).all()
)

check(
    "Order amounts are numeric",
    pd.api.types.is_numeric_dtype(orders["order_amount"])
)

status_counts = orders["status"].value_counts()

completed_count = status_counts.get("completed", 0)
cancelled_count = status_counts.get("cancelled", 0)
pending_count = status_counts.get("pending", 0)

print("\nOrder status counts:")
print(status_counts)

check("Completed orders = 11110", completed_count == 11110)
check("Cancelled orders = 1199", cancelled_count == 1199)
check("Pending orders = 0", pending_count == 0)

completed_orders = orders[
    orders["status"] == "completed"
]

cancelled_orders = orders[
    orders["status"] == "cancelled"
]

check(
    "No missing delivery times in completed orders",
    completed_orders["delivery_minutes"].notna().all()
)

check(
    "Cancelled orders have missing delivery times",
    cancelled_orders["delivery_minutes"].isna().all()
)

actual_revenue = completed_orders["order_amount"].sum()

print(f"\nCompleted order revenue: {actual_revenue:.2f}")

check(
    "Completed order revenue matches expected value",
    abs(actual_revenue - EXPECTED_REVENUE) < 0.01
)

# ==========================================
# 5. SQLITE DATABASE VALIDATION
# ==========================================

print("\n5. SQLITE DATABASE VALIDATION")

conn = sqlite3.connect(DATABASE_FILE)

tables = pd.read_sql_query(
    """
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
    """,
    conn
)["name"].tolist()

print("Database tables:", tables)

check(
    "Customers table exists",
    "customers" in tables
)

check(
    "Products table exists",
    "products" in tables
)

check(
    "Orders table exists",
    "orders" in tables
)

if all(table in tables for table in ["customers", "products", "orders"]):

    db_customers = pd.read_sql_query(
        "SELECT COUNT(*) AS total FROM customers", conn
    )["total"][0]

    db_products = pd.read_sql_query(
        "SELECT COUNT(*) AS total FROM products", conn
    )["total"][0]

    db_orders = pd.read_sql_query(
        "SELECT COUNT(*) AS total FROM orders", conn
    )["total"][0]

    check("Database customers = 1200", db_customers == 1200)
    check("Database products = 40", db_products == 40)
    check("Database orders = 12309", db_orders == 12309)

    db_revenue = pd.read_sql_query(
        """
        SELECT SUM(order_amount) AS revenue
        FROM orders
        WHERE status = 'completed'
        """,
        conn
    )["revenue"][0]

    print(f"Database completed revenue: {db_revenue:.2f}")

    check(
        "Database revenue matches expected value",
        db_revenue is not None
        and abs(db_revenue - EXPECTED_REVENUE) < 0.01
    )

else:
    print("[FAIL] Cannot validate row counts: required tables missing")
    failed += 1

conn.close()

# ==========================================
# 6. FINAL SUMMARY
# ==========================================

print("\n======================================")
print(" PART A VALIDATION SUMMARY")
print("======================================")

print("Checks passed:", passed)
print("Checks failed:", failed)

if failed == 0:
    print("\nALL CHECKPOINTS PASSED")
else:
    print("\nVALIDATION FAILED - PLEASE FIX THE ISSUES")

print("======================================")
