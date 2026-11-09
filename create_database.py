
import pandas as pd
import sqlite3
import os

# 1. Read cleaned datasets
customers = pd.read_csv(
    "data/clean/customers.csv",
    parse_dates=["signup_date"]
)

products = pd.read_csv("data/clean/products.csv")

orders = pd.read_csv(
    "data/clean/orders.csv",
    parse_dates=["order_ts"]
)

print("Customers loaded:", len(customers))
print("Products loaded:", len(products))
print("Orders loaded:", len(orders))

# 2. Create database directory
os.makedirs("data", exist_ok=True)

# 3. Connect to SQLite database
connection = sqlite3.connect("data/instamart.db")

# 4. Write datasets into database tables
customers.to_sql(
    "customers",
    connection,
    if_exists="replace",
    index=False
)

products.to_sql(
    "products",
    connection,
    if_exists="replace",
    index=False
)

orders.to_sql(
    "orders",
    connection,
    if_exists="replace",
    index=False
)

# 5. Verify tables
cursor = connection.cursor()

cursor.execute("""
SELECT name
FROM sqlite_master
WHERE type='table'
ORDER BY name
""")

print("\nTables created:")
for table in cursor.fetchall():
    print(table[0])

# 6. Verify row counts
for table in ["customers", "products", "orders"]:
    result = pd.read_sql_query(
        f"SELECT COUNT(*) AS total FROM {table}",
        connection
    )
    print(f"{table}: {result.iloc[0]['total']} rows")

connection.close()

print("\nSQLite database created successfully!")
