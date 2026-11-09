
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load RFM data
rfm = pd.read_csv("outputs/rfm_analysis.csv")

print("RFM dataset shape:", rfm.shape)

# 2. Select features for clustering
features = ["recency", "frequency", "monetary"]
X = rfm[features]

# 3. Standardise the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeatures standardised successfully!")

# 4. Test K-Means for K = 2 to 8
results = []

for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    results.append({
        "K": k,
        "Inertia": inertia,
        "Silhouette Score": silhouette
    })

# 5. Display results
results_df = pd.DataFrame(results)

print("\n===== K-MEANS EVALUATION =====")
print(results_df.to_string(index=False))

# 6. Save evaluation results
results_df.to_csv(
    "outputs/kmeans_evaluation.csv",
    index=False
)

print("\nK-Means evaluation saved successfully!")
