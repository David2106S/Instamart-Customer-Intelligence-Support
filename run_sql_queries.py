
import sqlite3
import pandas as pd

# Connect to SQLite database
connection = sqlite3.connect("data/instamart.db")

# Read SQL file
with open("sql/queries.sql", "r") as file:
    sql_script = file.read()

# Split SQL statements
queries = sql_script.split(";")

# Execute each query
for index, query in enumerate(queries, start=1):

    # Remove SQL comment lines
    clean_query = "\n".join(
        line for line in query.splitlines()
        if not line.strip().startswith("--")
    ).strip()

    if not clean_query:
        continue

    print(f"\n{'=' * 50}")
    print(f"QUERY {index}")
    print("=" * 50)

    try:
        result = pd.read_sql_query(clean_query, connection)

        print(result.head(10))
        print("\nTotal rows returned:", len(result))

    except Exception as error:
        print("Query failed:", error)

connection.close()

print("\nAll SQL queries executed!")
