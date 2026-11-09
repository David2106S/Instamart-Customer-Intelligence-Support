
import pandas as pd
import math

# 1. Load ticket dataset
df = pd.read_csv("data/raw/tickets.csv")

print("Total tickets:", len(df))

# 2. Estimate tokens
def estimate_tokens(text):
    if pd.isna(text) or not str(text).strip():
        return 0

    # Approximate token count
    return math.ceil(len(str(text).split()) * 1.3)

df["estimated_tokens"] = df["text"].apply(estimate_tokens)

# 3. Identify empty tickets
empty_mask = (
    df["text"].isna() |
    df["text"].astype(str).str.strip().eq("")
)

# 4. Identify long tickets
long_mask = (
    ~empty_mask &
    (df["estimated_tokens"] > 400)
)

# 5. Assign ticket status
df["status"] = "eligible"

df.loc[empty_mask, "status"] = "skipped_empty"
df.loc[long_mask, "status"] = "flagged_long"

# 6. Display results
print("\n===== TICKET GUARDRAIL RESULTS =====")

print("\nStatus distribution:")
print(df["status"].value_counts())

print("\nEmpty tickets:", empty_mask.sum())
print("Long tickets:", long_mask.sum())
print("Eligible tickets:", (df["status"] == "eligible").sum())

# 7. Validate expected counts
assert empty_mask.sum() == 4, "Expected 4 empty tickets"
assert long_mask.sum() == 4, "Expected 4 long tickets"
assert (df["status"] == "eligible").sum() == 156, "Expected 156 eligible tickets"

# 8. Save output
df.to_csv("outputs/ticket_guardrails.csv", index=False)

print("\nALL TICKET GUARDRAIL CHECKPOINTS PASSED!")
print("Output: outputs/ticket_guardrails.csv")
