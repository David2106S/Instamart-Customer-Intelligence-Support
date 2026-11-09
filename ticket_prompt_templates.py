
import pandas as pd

# 1. Load redacted ticket dataset
df = pd.read_csv("outputs/tickets_redacted.csv")

# 2. Zero-shot prompt template
def create_zero_shot_prompt(ticket_text):

    prompt = f"""
CLASSIFY the following customer support ticket.

Choose exactly one category:
- billing
- delivery
- technical
- fraud

Return ONLY valid JSON in this format:
{{"category":"...","reason":"..."}}

Ticket:
{ticket_text}
"""

    return prompt


# 3. Few-shot prompt template
def create_few_shot_prompt(ticket_text):

    prompt = f"""
CLASSIFY the following customer support ticket.

Choose exactly one category:
- billing
- delivery
- technical
- fraud

EXAMPLES:

Ticket: I was charged twice for my order.
Output: {{"category":"billing","reason":"Duplicate payment"}}

Ticket: My delivery has not arrived yet.
Output: {{"category":"delivery","reason":"Delayed delivery"}}

Ticket: Someone used my card without permission.
Output: {{"category":"fraud","reason":"Unauthorized transaction"}}

Return ONLY valid JSON in this format:
{{"category":"...","reason":"..."}}

Ticket:
{ticket_text}
"""

    return prompt


# 4. Validate prompt requirements
zero_prompt = create_zero_shot_prompt("My invoice shows an incorrect amount.")
few_prompt = create_few_shot_prompt("My invoice shows an incorrect amount.")

print("===== PROMPT VALIDATION =====")

assert "CLASSIFY" in zero_prompt
assert "EXAMPLES:" not in zero_prompt

assert "CLASSIFY" in few_prompt
assert "EXAMPLES:" in few_prompt

for category in ["billing", "delivery", "technical", "fraud"]:
    assert category in zero_prompt
    assert category in few_prompt

assert '{"category":"...","reason":"..."}' in zero_prompt
assert '{"category":"...","reason":"..."}' in few_prompt

assert few_prompt.count("Output:") >= 3
assert '"category":"fraud"' in few_prompt

print("Zero-shot prompt: PASSED")
print("Few-shot prompt: PASSED")
print("All four categories: PASSED")
print("JSON output format: PASSED")
print("Few-shot examples including fraud: PASSED")


# 5. Prepare prompts for eligible tickets only
eligible_df = df[df["status"] == "eligible"].copy()

eligible_df["zero_shot_prompt"] = eligible_df[
    "redacted_text"
].apply(create_zero_shot_prompt)

eligible_df["few_shot_prompt"] = eligible_df[
    "redacted_text"
].apply(create_few_shot_prompt)

# 6. Verify prompts use redacted text
assert eligible_df["zero_shot_prompt"].notna().all()
assert eligible_df["few_shot_prompt"].notna().all()

assert len(eligible_df) == 156

# 7. Save prompt dataset
eligible_df.to_csv(
    "outputs/ticket_prompts.csv",
    index=False
)

print("\nEligible tickets:", len(eligible_df))
print("Prompt dataset saved successfully!")
print("Output: outputs/ticket_prompts.csv")
