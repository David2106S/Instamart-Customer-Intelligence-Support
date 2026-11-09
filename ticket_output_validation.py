
import pandas as pd
import json
import re

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

CATEGORIES = ["billing", "delivery", "technical", "fraud"]

# --------------------------------------------------
# 2. VALIDATE RAW LLM RESPONSES
# --------------------------------------------------

def validate_responses(raw_file, output_file):

    raw_df = pd.read_csv(raw_file)

    # Load original tickets to obtain actual categories
    tickets_df = pd.read_csv("data/raw/tickets.csv")

    # Join actual category with LLM response
    df = raw_df.merge(
        tickets_df[["ticket_id", "true_category"]],
        on="ticket_id",
        how="left"
    )

    statuses = []
    predicted_categories = []

    for response in df["raw_response"]:

        category = None
        status = "flagged_invalid_output"

        # Attempt JSON parsing first
        try:
            result = json.loads(str(response))

            # Valid JSON with an allowed category
            if (
                isinstance(result, dict)
                and result.get("category") in CATEGORIES
            ):
                category = result["category"]
                status = "classified_json"

            # Valid JSON but invalid category:
            # Keep it flagged. Do not use keyword fallback.

        except (json.JSONDecodeError, TypeError, ValueError):

            # Non-JSON response: use keyword fallback
            words = set(
                re.findall(r"[a-z]+", str(response).lower())
            )

            matched_categories = [
                cat for cat in CATEGORIES if cat in words
            ]

            if len(matched_categories) == 1:
                category = matched_categories[0]
                status = "classified_keyword"

        statuses.append(status)
        predicted_categories.append(category)

    df["status"] = statuses
    df["predicted_category"] = predicted_categories

    df.to_csv(output_file, index=False)

    return df


# --------------------------------------------------
# 3. VALIDATE ZERO-SHOT AND FEW-SHOT
# --------------------------------------------------

zero_df = validate_responses(
    "outputs/triage_raw_zero_shot.csv",
    "outputs/triage_validated_zero_shot.csv"
)

few_df = validate_responses(
    "outputs/triage_raw_few_shot.csv",
    "outputs/triage_validated_few_shot.csv"
)


# --------------------------------------------------
# 4. DISPLAY VALIDATION COUNTS
# --------------------------------------------------

def display_validation(name, df):

    print(f"\n===== {name} VALIDATION =====")

    print("\nStatus distribution:")
    print(df["status"].value_counts())

    classified = df[
        df["status"].isin([
            "classified_json",
            "classified_keyword"
        ])
    ]

    print("\nTotal tickets:", len(df))
    print("Classified tickets:", len(classified))
    print("Invalid tickets:", (df["status"] == "flagged_invalid_output").sum())

    print("\nPredicted category distribution:")
    print(classified["predicted_category"].value_counts())

    accuracy = accuracy_score(
        classified["true_category"],
        classified["predicted_category"]
    )

    print("\nAccuracy:", round(accuracy, 4))

    return classified, accuracy


zero_classified, zero_accuracy = display_validation(
    "ZERO-SHOT",
    zero_df
)

few_classified, few_accuracy = display_validation(
    "FEW-SHOT",
    few_df
)


# --------------------------------------------------
# 5. ZERO-SHOT CONFUSION MATRIX
# --------------------------------------------------

print("\n===== ZERO-SHOT CONFUSION MATRIX =====")

cm = confusion_matrix(
    zero_classified["true_category"],
    zero_classified["predicted_category"],
    labels=CATEGORIES
)

cm_df = pd.DataFrame(
    cm,
    index=CATEGORIES,
    columns=CATEGORIES
)

print(cm_df)

cm_df.to_csv("outputs/zero_shot_confusion_matrix.csv")


# --------------------------------------------------
# 6. CLASSIFICATION REPORTS
# --------------------------------------------------

print("\n===== ZERO-SHOT CLASSIFICATION REPORT =====")

print(classification_report(
    zero_classified["true_category"],
    zero_classified["predicted_category"],
    labels=CATEGORIES,
    zero_division=0
))

print("\n===== FEW-SHOT CLASSIFICATION REPORT =====")

print(classification_report(
    few_classified["true_category"],
    few_classified["predicted_category"],
    labels=CATEGORIES,
    zero_division=0
))


# --------------------------------------------------
# 7. TOKEN USAGE COMPARISON
# --------------------------------------------------

comparison = pd.DataFrame([
    {
        "prompt_type": "Zero-shot",
        "total_tickets": len(zero_df),
        "classified_tickets": len(zero_classified),
        "invalid_tickets": len(zero_df) - len(zero_classified),
        "accuracy": zero_accuracy,
        "total_tokens": zero_df["tokens_used"].sum(),
        "tokens_per_classified_ticket":
            zero_df["tokens_used"].sum() / len(zero_classified)
    },
    {
        "prompt_type": "Few-shot",
        "total_tickets": len(few_df),
        "classified_tickets": len(few_classified),
        "invalid_tickets": len(few_df) - len(few_classified),
        "accuracy": few_accuracy,
        "total_tokens": few_df["tokens_used"].sum(),
        "tokens_per_classified_ticket":
            few_df["tokens_used"].sum() / len(few_classified)
    }
])

print("\n===== FINAL ZERO-SHOT VS FEW-SHOT COMPARISON =====")

print(comparison.round(4).to_string(index=False))

comparison.to_csv(
    "outputs/triage_prompt_comparison.csv",
    index=False
)


# --------------------------------------------------
# 8. CHECK EXPECTED ASSIGNMENT RESULTS
# --------------------------------------------------

print("\n===== ASSIGNMENT CHECKPOINTS =====")

for name, df in [
    ("ZERO-SHOT", zero_df),
    ("FEW-SHOT", few_df)
]:

    counts = df["status"].value_counts()

    json_count = counts.get("classified_json", 0)
    keyword_count = counts.get("classified_keyword", 0)
    invalid_count = counts.get("flagged_invalid_output", 0)

    print(f"\n{name}")
    print("JSON classifications:", json_count)
    print("Keyword classifications:", keyword_count)
    print("Invalid outputs:", invalid_count)

    assert json_count == 135, f"{name}: JSON count mismatch"
    assert keyword_count == 12, f"{name}: Keyword count mismatch"
    assert invalid_count == 9, f"{name}: Invalid count mismatch"

    assert len(df) == 156, f"{name}: Ticket count mismatch"

print("\nALL TICKET OUTPUT VALIDATION CHECKPOINTS PASSED!")

print("\nGenerated files:")
print("outputs/triage_validated_zero_shot.csv")
print("outputs/triage_validated_few_shot.csv")
print("outputs/zero_shot_confusion_matrix.csv")
print("outputs/triage_prompt_comparison.csv")
