
import pandas as pd
import mock_llm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. LOAD POLICY CHUNKS
df = pd.read_csv("outputs/policy_chunks.csv")

# 2. CREATE TF-IDF VECTORS
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["chunk_text"])


# 3. RETRIEVAL FUNCTION

def retrieve(query, k=3):

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    )[0]

    top_indices = scores.argsort()[::-1][:k]

    results = []

    for index in top_indices:

        results.append({
            "index": int(index),
            "source": df.iloc[index]["source"],
            "score": float(scores[index]),
            "text": df.iloc[index]["chunk_text"]
        })

    return results


def merge_retrieved_chunks(results):

    # Group retrieved chunks by source document
    grouped = {}

    for item in results:
        grouped.setdefault(item["source"], []).append(item)

    merged_context = []

    for source, items in grouped.items():

        # Restore original document order
        items.sort(key=lambda x: x["index"])

        combined_words = []

        for item in items:

            words = item["text"].split()

            # Find overlap between previous chunk and current chunk
            overlap = 0

            max_overlap = min(10, len(combined_words), len(words))

            for size in range(max_overlap, 0, -1):

                if combined_words[-size:] == words[:size]:
                    overlap = size
                    break

            combined_words.extend(words[overlap:])

        merged_context.append(
            f"[{source}]\n{' '.join(combined_words)}"
        )

    return "\n".join(merged_context)



# 4. RAG ANSWER FUNCTION
def answer(question):

    # Retrieve top 3 policy chunks
    results = retrieve(question, k=3)

    # Build context using retrieved documents
    context = merge_retrieved_chunks(results)

    # System prompt
    system_prompt = """
    POLICY ASSISTANT

    Answer the user's question using ONLY the
    policy context provided.

    If the answer is not available in the context,
    respond that you could not find the information
    in the policy documents.
    """

    # Exact user message structure required
    user_message = (
        f"CONTEXT:\n{context}\n"
        f"QUESTION:\n{question}"
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    # Call offline mock LLM
    response = mock_llm.chat(messages)

    generated_answer = response.choices[0].message.content

    # Return answer and retrieved sources
    sources = list(dict.fromkeys(
        item["source"] for item in results
    ))

    return generated_answer, sources


# 5. TEST THE RAG ASSISTANT
if __name__ == "__main__":

    question = (
        "What happens if a completed order "
        "is delivered in more than 45 minutes?"
    )

    result, sources = answer(question)

    print("===== POLICY RAG ASSISTANT =====")
    print("\nQuestion:", question)

    print("\nAnswer:")
    print(result)

    print("\nRetrieved Sources:")
    for source in sources:
        print("-", source)
