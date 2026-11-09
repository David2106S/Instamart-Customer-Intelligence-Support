
import pandas as pd

# --------------------------------------------------
# 1. LOAD INPUT DATASETS
# --------------------------------------------------

guardrails = pd.read_csv("outputs/ticket_guardrails.csv")

classification = pd.read_csv(
    "outputs/triage_validated_few_shot.csv"
)

segments = pd.read_csv("outputs/customer_segments.csv")

print("Guardrail records:", len(guardrails))
print("Classification records:", len(classification))
print("Customer segment records:", len(segments))


# --------------------------------------------------
# 2. PREPARE CLASSIFICATION DATA
# --------------------------------------------------

classification = classification[
    [
        "ticket_id",
        "status",
        "predicted_category",
        "tokens_used"
    ]
].rename(columns={
    "status": "classification_status",
    "predicted_category": "category"
})


# --------------------------------------------------
# 3. MERGE GUARDRAILS WITH CLASSIFICATION
# --------------------------------------------------

final_df = guardrails.merge(
    classification,
    on="ticket_id",
    how="left"
)

# Preserve skipped ticket statuses.
# For eligible tickets, use the LLM classification status.

final_df["status"] = final_df["classification_status"].where(
    final_df["status"] == "eligible",
    final_df["status"]
)

# If an eligible ticket has no classification result,
# send it for human review.

missing_classification = (
    final_df["status"].eq("eligible")
    & final_df["classification_status"].isna()
)

final_df.loc[
    missing_classification, "status"
] = "flagged_invalid_output"

final_df["category"] = final_df["category"].fillna("")


# --------------------------------------------------
# 4. MERGE CUSTOMER SEGMENTS
# --------------------------------------------------

segment_columns = [
    "customer_id",
    "segment_name",
    "is_premium"
]

final_df = final_df.merge(
    segments[segment_columns],
    on="customer_id",
    how="left"
)

final_df["segment_name"] = final_df["segment_name"].fillna("")

# A customer without a matching segment is not premium.

premium_mask = (
    final_df["is_premium"].astype(str).str.lower().eq("true")
    | final_df["segment_name"].eq("Premium Customers")
)


# --------------------------------------------------
# 5. APPLY PRIORITY AND ROUTING RULES
# --------------------------------------------------

def assign_routing(row):

    status = row["status"]
    category = row["category"]

    # Guardrail failures and invalid outputs
    if status in [
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted"
    ]:
        return pd.Series(["NORMAL", "human_review"])

    # Fraud tickets receive highest routing priority
    if category == "fraud":
        return pd.Series(["HIGH", "security_team"])

    # Premium customer tickets
    if premium_mask.loc[row.name]:
        return pd.Series(["HIGH", "premium_desk"])

    # Remaining valid tickets
    return pd.Series(["NORMAL", "support_queue"])


final_df[["priority", "route"]] = final_df.apply(
    assign_routing,
    axis=1
)


# --------------------------------------------------
# 6. SELECT REQUIRED OUTPUT COLUMNS
# --------------------------------------------------

output_columns = [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used"
]

final_output = final_df[output_columns].copy()

final_output["tokens_used"] = (
    final_output["tokens_used"].fillna(0).astype(int)
)


# --------------------------------------------------
# 7. SAVE FINAL OUTPUT
# --------------------------------------------------

output_file = "outputs/triage_results.csv"

final_output.to_csv(output_file, index=False)


# --------------------------------------------------
# 8. VALIDATION
# --------------------------------------------------

print("\n===== FINAL TICKET ROUTING RESULTS =====")

print("Total tickets:", len(final_output))

print("\nStatus distribution:")
print(final_output["status"].value_counts())

print("\nPriority distribution:")
print(final_output["priority"].value_counts())

print("\nRouting distribution:")
print(final_output["route"].value_counts())

print("\nSample output:")
print(final_output.head(10).to_string(index=False))

assert len(final_output) == 164, "Expected exactly 164 tickets"

assert not final_output["ticket_id"].duplicated().any(), (
    "Duplicate ticket IDs found"
)

assert final_output["route"].notna().all()
assert final_output["priority"].notna().all()

print("\nALL FINAL TICKET ROUTING CHECKPOINTS PASSED!")

print("Output:", output_file)
