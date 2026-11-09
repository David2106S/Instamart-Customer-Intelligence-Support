
import pandas as pd
import re

# 1. Load guardrail output
df = pd.read_csv("outputs/ticket_guardrails.csv")

# 2. Define PII patterns
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'

phone_pattern = r'\b\d{10}\b'

order_pattern = r'\bORD-\d{6}\b'

# 3. Redact PII and count replacements
email_count = 0
phone_count = 0
order_count = 0

def redact_pii(text):

    global email_count, phone_count, order_count

    if pd.isna(text):
        return text

    text = str(text)

    text, emails = re.subn(
        email_pattern, "[EMAIL]", text
    )

    text, phones = re.subn(
        phone_pattern, "[PHONE]", text
    )

    text, orders = re.subn(
        order_pattern, "[ORDER_ID]", text
    )

    email_count += emails
    phone_count += phones
    order_count += orders

    return text

# 4. Apply redaction
df["redacted_text"] = df["text"].apply(redact_pii)

# 5. Display results
print("===== PII REDACTION RESULTS =====")

print("Email replacements:", email_count)
print("Phone replacements:", phone_count)
print("Order ID replacements:", order_count)

total = email_count + phone_count + order_count

print("Total replacements:", total)

# 6. Validate expected counts
assert email_count == 55, "Email count mismatch"
assert phone_count == 91, "Phone count mismatch"
assert order_count == 132, "Order ID count mismatch"

# 7. Verify no original PII remains
remaining_emails = df["redacted_text"].str.contains(
    email_pattern, na=False
).sum()

remaining_phones = df["redacted_text"].str.contains(
    phone_pattern, na=False
).sum()

remaining_orders = df["redacted_text"].str.contains(
    order_pattern, na=False
).sum()

print("\nRemaining emails:", remaining_emails)
print("Remaining phone numbers:", remaining_phones)
print("Remaining order IDs:", remaining_orders)

assert remaining_emails == 0
assert remaining_phones == 0
assert remaining_orders == 0

# 8. Save redacted dataset
df.to_csv(
    "outputs/tickets_redacted.csv",
    index=False
)

print("\nALL PII REDACTION CHECKPOINTS PASSED!")
print("Output: outputs/tickets_redacted.csv")
