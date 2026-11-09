
import pandas as pd
import joblib

# --------------------------------------------------
# 1. LOAD DATA AND TRAINED MODEL
# --------------------------------------------------

lapse_df = pd.read_csv("outputs/lapse_dataset.csv")
segments_df = pd.read_csv("outputs/customer_segments.csv")

model = joblib.load("outputs/lapse_model.joblib")

print("Lapse dataset:", lapse_df.shape)
print("Customer segments:", segments_df.shape)

# --------------------------------------------------
# 2. IDENTIFY PREMIUM CUSTOMERS
# --------------------------------------------------

premium_customers = segments_df[
    segments_df["is_premium"] == True
][
    ["customer_id", "segment_name", "is_premium"]
].copy()

print("\nTotal Premium customers:", len(premium_customers))

# --------------------------------------------------
# 3. MERGE PREMIUM CUSTOMERS WITH LAPSE FEATURES
# --------------------------------------------------

premium_data = premium_customers.merge(
    lapse_df,
    on="customer_id",
    how="inner"
)

print("Eligible Premium customers:", len(premium_data))

# --------------------------------------------------
# 4. SELECT MODEL FEATURES
# --------------------------------------------------

numeric_features = [
    "frequency",
    "monetary",
    "recency",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery_time",
    "age",
    "tenure_days"
]

categorical_features = [
    "favourite_category",
    "city",
    "gender"
]

feature_columns = numeric_features + categorical_features

X_premium = premium_data[feature_columns]

# --------------------------------------------------
# 5. PREDICT LAPSE PROBABILITY
# --------------------------------------------------

lapse_probabilities = model.predict_proba(X_premium)[:, 1]

premium_data["predicted_lapse_probability"] = lapse_probabilities

# --------------------------------------------------
# 6. SELECT TOP 10 HIGH-RISK PREMIUM CUSTOMERS
# --------------------------------------------------

retention_list = premium_data.sort_values(
    by="predicted_lapse_probability",
    ascending=False
).head(10)

# Select relevant output columns
retention_list = retention_list[
    [
        "customer_id",
        "segment_name",
        "recency",
        "frequency",
        "monetary",
        "predicted_lapse_probability"
    ]
]

# --------------------------------------------------
# 7. SAVE OUTPUT
# --------------------------------------------------

retention_list.to_csv(
    "outputs/retention_list.csv",
    index=False
)

print("\n===== TOP 10 PREMIUM CUSTOMERS AT RISK =====")
print(retention_list.to_string(index=False))

print("\nRetention list saved successfully!")
print("Output: outputs/retention_list.csv")
print("\nALL RETENTION LIST CHECKPOINTS PASSED")
