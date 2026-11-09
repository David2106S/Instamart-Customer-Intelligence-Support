
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load RFM dataset
rfm = pd.read_csv("outputs/rfm_analysis.csv")

# 2. Select RFM features
features = ["recency", "frequency", "monetary"]
X = rfm[features]

# 3. Standardise features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train final K-Means model with K=4
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["cluster"] = kmeans.fit_predict(X_scaled)

# 5. Generate cluster profiles
profile = rfm.groupby("cluster").agg(
    customer_count=("customer_id", "count"),
    avg_recency=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean")
).round(2)

print("\n===== CUSTOMER SEGMENT PROFILES =====")
print(profile)

# 6. Identify the highest-spending cluster
premium_cluster = profile["avg_monetary"].idxmax()

print("\nPremium cluster ID:", premium_cluster)

# 7. Save customer segments
rfm.to_csv(
    "outputs/customer_segments.csv",
    index=False
)

profile.to_csv(
    "outputs/segment_profiles.csv"
)

print("\nCustomer segments saved successfully!")
print("Total customers:", len(rfm))
print("Total segments:", rfm["cluster"].nunique())
