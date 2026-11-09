
import os
import pandas as pd

POLICY_FOLDER = "data/policies"
CHUNK_SIZE = 40
OVERLAP = 10

chunks = []

for filename in os.listdir(POLICY_FOLDER):

    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(POLICY_FOLDER, filename)

    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()

    words = text.split()

    start = 0

    while start < len(words):

        end = start + CHUNK_SIZE

        chunk_words = words[start:end]

        chunks.append({
            "source": filename,
            "chunk_text": " ".join(chunk_words)
        })

        if end >= len(words):
            break

        start = end - OVERLAP

df = pd.DataFrame(chunks)

os.makedirs("outputs", exist_ok=True)

df.to_csv("outputs/policy_chunks.csv", index=False)

print("===== POLICY CHUNKING RESULTS =====")
print("Total chunks:", len(df))
print("\nChunks by document:")
print(df["source"].value_counts())

print("\nOutput: outputs/policy_chunks.csv")
