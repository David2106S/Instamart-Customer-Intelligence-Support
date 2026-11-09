
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load customer segmentation results
df = pd.read_csv("outputs/customer_segments.csv")

# 2. Assign meaningful segment names
segment_names = {
    0: "Regular Customers",
    1: "Premium Customers",
    2: "Loyal Customers",
    3: "At-Risk Customers"
}

df["segment_name"] = df["cluster"].map(segment_names)

# 3. Identify Premium customers
df["is_premium"] = df["cluster"] == 1

# Save updated customer segmentation
df.to_csv("outputs/customer_segments.csv", index=False)

# 4. Select RFM features
features = ["recency", "frequency", "monetary"]
X = df[features]

# 5. Standardise features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Apply PCA with two components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

# 7. Add PCA coordinates
df["PCA1"] = pca_result[:, 0]
df["PCA2"] = pca_result[:, 1]

# 8. Display explained variance
print("\n===== PCA RESULTS =====")

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

print(
    "\nTotal explained variance:",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)

# 9. Save PCA results
df.to_csv("outputs/pca_customer_segments.csv", index=False)

# 10. Plot customer segments
plt.figure(figsize=(10, 6))

for cluster in sorted(df["cluster"].unique()):
    cluster_data = df[df["cluster"] == cluster]

    plt.scatter(
        cluster_data["PCA1"],
        cluster_data["PCA2"],
        label=segment_names[cluster],
        alpha=0.6
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Customer Segmentation using PCA")
plt.legend()
plt.grid(True)

plt.savefig("outputs/customer_segments_pca.png")
plt.show()

print("\nPCA analysis completed successfully!")
