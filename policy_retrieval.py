
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load policy chunks
df = pd.read_csv("outputs/policy_chunks.csv")

# 2. Create TF-IDF vectors
vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(df["chunk_text"])

# 3. Build retrieval function
def retrieve(query, k=3):

    query_vector = vectorizer.transform([query])

    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    )[0]

    top_indices = similarity_scores.argsort()[::-1][:k]

    results = []

    for index in top_indices:

        results.append({
            "source": df.iloc[index]["source"],
            "score": round(float(similarity_scores[index]), 4),
            "text": df.iloc[index]["chunk_text"]
        })

    return results


# 4. Test retrieval
query = "refund policy late delivery"

results = retrieve(query, k=3)

print("===== POLICY RETRIEVAL RESULTS =====")
print("Query:", query)

for i, result in enumerate(results, start=1):

    print(f"\nResult {i}")
    print("Source:", result["source"])
    print("Similarity Score:", result["score"])
    print("Text:", result["text"])
