
import os
import pandas as pd
import policy_rag

from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------
# 1. LOAD EVALUATION QUESTIONS
# --------------------------------------------------

eval_df = pd.read_csv("data/raw/policy_eval.csv")

print("Total evaluation questions:", len(eval_df))
print("Columns:", eval_df.columns.tolist())


# --------------------------------------------------
# 2. EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_rag(data, experiment_name):

    results = []

    for _, row in data.iterrows():

        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]

        # Retrieve top 3 chunks
        retrieved = policy_rag.retrieve(question, k=3)

        retrieved_sources = [
            item["source"] for item in retrieved
        ]

        # Check expected document
        if expected_doc == "NONE":
            document_match = True
        else:
            document_match = expected_doc in retrieved_sources

        # Generate answer
        generated_answer, sources = policy_rag.answer(question)

        # Validate expected fact or unsupported fallback
        if expected_doc == "NONE":

            fallback_match = generated_answer.startswith(
                "I could not find"
            )

            fact_match = fallback_match

        else:

            fact_match = (
                isinstance(expected_fact, str)
                and expected_fact in generated_answer
            )

        results.append({
            "question": question,
            "expected_doc": expected_doc,
            "retrieved_sources": ", ".join(retrieved_sources),
            "generated_answer": generated_answer,
            "document_match": document_match,
            "fact_or_fallback_match": fact_match
        })

    result_df = pd.DataFrame(results)

    total = len(result_df)

    if experiment_name == "OVERLAP":

        output_file = "outputs/rag_eval_overlap.csv"

    else:

        output_file = "outputs/rag_eval_zero_overlap.csv"

    result_df.to_csv(output_file, index=False)

    answerable = result_df[
        result_df["expected_doc"] != "NONE"
    ]

    unsupported = result_df[
        result_df["expected_doc"] == "NONE"
    ]

    print(f"\n===== {experiment_name} EVALUATION =====")

    print("Total questions:", total)

    print(
        "Expected document matches:",
        int(answerable["document_match"].sum()),
        "/",
        len(answerable)
    )

    print(
        "Expected fact matches:",
        int(answerable["fact_or_fallback_match"].sum()),
        "/",
        len(answerable)
    )

    print(
        "Unsupported fallback matches:",
        int(unsupported["fact_or_fallback_match"].sum()),
        "/",
        len(unsupported)
    )

    print("\nFailed answerable questions:")

    failures = answerable[
        ~answerable["fact_or_fallback_match"]
    ]

    if failures.empty:
        print("None")
    else:
        print(
            failures[
                ["question", "generated_answer"]
            ].to_string(index=False)
        )

    print("\nSaved:", output_file)

    return result_df


# --------------------------------------------------
# 3. EXPERIMENT 1: 40 WORDS, 10 WORD OVERLAP
# --------------------------------------------------

print("\nRunning evaluation with 10-word overlap...")

# policy_rag already contains the original 10-word overlap chunks.
overlap_results = evaluate_rag(
    eval_df,
    "OVERLAP"
)


# --------------------------------------------------
# 4. EXPERIMENT 2: ZERO OVERLAP
# --------------------------------------------------

print("\nRebuilding policy chunks with zero overlap...")

zero_chunks = []

policy_folder = "data/policies"

for filename in os.listdir(policy_folder):

    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(policy_folder, filename)

    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()

    words = text.split()

    for start in range(0, len(words), 40):

        chunk_words = words[start:start + 40]

        zero_chunks.append({
            "source": filename,
            "chunk_text": " ".join(chunk_words)
        })


zero_df = pd.DataFrame(zero_chunks)

# Replace the retrieval system's data with zero-overlap chunks.
policy_rag.df = zero_df

policy_rag.vectorizer = TfidfVectorizer()

policy_rag.tfidf_matrix = (
    policy_rag.vectorizer.fit_transform(
        zero_df["chunk_text"]
    )
)

print("Zero-overlap chunk count:", len(zero_df))


# --------------------------------------------------
# 5. EVALUATE ZERO-OVERLAP RESULTS
# --------------------------------------------------

zero_results = evaluate_rag(
    eval_df,
    "ZERO_OVERLAP"
)


# --------------------------------------------------
# 6. FINAL COMPARISON
# --------------------------------------------------

print("\n===== FINAL RAG EVALUATION SUMMARY =====")

print("\n10-word overlap:")
print(
    "Document matches:",
    overlap_results[
        overlap_results["expected_doc"] != "NONE"
    ]["document_match"].sum(),
    "/ 12"
)

print(
    "Fact matches and fallbacks:",
    overlap_results[
        overlap_results["expected_doc"] != "NONE"
    ]["fact_or_fallback_match"].sum(),
    "/ 12"
)

print(
    "Unsupported fallbacks:",
    overlap_results[
        overlap_results["expected_doc"] == "NONE"
    ]["fact_or_fallback_match"].sum(),
    "/ 2"
)

print("\nZero overlap:")
print(
    "Document matches:",
    zero_results[
        zero_results["expected_doc"] != "NONE"
    ]["document_match"].sum(),
    "/ 12"
)

print(
    "Fact matches and fallbacks:",
    zero_results[
        zero_results["expected_doc"] != "NONE"
    ]["fact_or_fallback_match"].sum(),
    "/ 12"
)

print(
    "Unsupported fallbacks:",
    zero_results[
        zero_results["expected_doc"] == "NONE"
    ]["fact_or_fallback_match"].sum(),
    "/ 2"
)

print("\nRAG EVALUATION COMPLETED!")
