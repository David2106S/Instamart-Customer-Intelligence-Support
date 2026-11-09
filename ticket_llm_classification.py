
import pandas as pd
import mock_llm
import random
import time
import re

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("outputs/tickets_redacted.csv")

eligible_df = df[df["status"] == "eligible"].copy()

print("Eligible tickets:", len(eligible_df))


# --------------------------------------------------
# 2. DEFINE SYSTEM PROMPTS
# --------------------------------------------------

zero_shot_system = """
CLASSIFY customer support tickets.

Choose exactly one category:
billing, delivery, technical, fraud.

Return ONLY valid JSON:
{"category":"...","reason":"..."}
"""

few_shot_system = """
CLASSIFY customer support tickets.

Choose exactly one category:
billing, delivery, technical, fraud.

EXAMPLES:

Ticket: I was charged twice for my order.
Output: {"category":"billing","reason":"Duplicate payment"}

Ticket: My delivery has not arrived.
Output: {"category":"delivery","reason":"Delayed delivery"}

Ticket: Someone used my card without permission.
Output: {"category":"fraud","reason":"Unauthorized transaction"}

Return ONLY valid JSON:
{"category":"...","reason":"..."}
"""


# --------------------------------------------------
# 3. RETRY CONFIGURATION
# --------------------------------------------------

BASE_DELAY = 0.05
MAX_RETRIES = 3

retry_logs = []


# --------------------------------------------------
# 4. LLM CALL WITH RETRIES
# --------------------------------------------------

def call_llm_with_retry(ticket_id, messages):

    for attempt in range(MAX_RETRIES + 1):

        try:

            response = mock_llm.chat(messages)

            return response

        except mock_llm.TransientLLMError:

            if attempt == MAX_RETRIES:
                raise

            retry_number = attempt + 1

            wait_time = (
                BASE_DELAY
                * (2 ** attempt)
                * random.uniform(0.9, 1.15)
            )

            retry_logs.append({
                "ticket_id": ticket_id,
                "retry_number": retry_number,
                "wait_time": round(wait_time, 4)
            })

            print(
                f"Retry: Ticket {ticket_id}, "
                f"Attempt {retry_number}, "
                f"Wait {wait_time:.4f}s"
            )

            time.sleep(wait_time)


# --------------------------------------------------
# 5. RUN CLASSIFICATION
# --------------------------------------------------

def run_classification(system_prompt, run_name):

    results = []

    total_tokens = 0

    # Reset immediately before each independent run
    mock_llm.reset()

    print(f"\n===== {run_name} STARTED =====")

    for _, row in eligible_df.iterrows():

        ticket_id = row["ticket_id"]
        ticket_text = str(row["redacted_text"])

        # Verify no PII remains in ticket text
        pii_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            r'\b\d{10}\b',
            r'\bORD-\d{6}\b'
        ]

        for pattern in pii_patterns:
            assert not re.search(pattern, ticket_text), (
                f"PII detected in ticket {ticket_id}"
            )

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Ticket:\n{ticket_text}"
            }
        ]

        response = call_llm_with_retry(
            ticket_id,
            messages
        )

        answer = response.choices[0].message.content

        tokens = response.usage.total_tokens

        total_tokens += tokens

        results.append({
            "ticket_id": ticket_id,
            "raw_response": answer,
            "tokens_used": tokens
        })

    result_df = pd.DataFrame(results)

    output_file = (
        "outputs/triage_raw_zero_shot.csv"
        if run_name == "ZERO-SHOT"
        else "outputs/triage_raw_few_shot.csv"
    )

    result_df.to_csv(output_file, index=False)

    print(f"\n===== {run_name} RESULTS =====")
    print("Tickets processed:", len(result_df))
    print("Total tokens:", total_tokens)
    print("Total retries:", len(retry_logs))
    print("Output:", output_file)

    return result_df, total_tokens


# --------------------------------------------------
# 6. RUN ZERO-SHOT CLASSIFICATION
# --------------------------------------------------

retry_logs.clear()

zero_results, zero_tokens = run_classification(
    zero_shot_system,
    "ZERO-SHOT"
)

zero_retry_count = len(retry_logs)

zero_retry_logs = pd.DataFrame(retry_logs)

zero_retry_logs.to_csv(
    "outputs/zero_shot_retry_log.csv",
    index=False
)


# --------------------------------------------------
# 7. RUN FEW-SHOT CLASSIFICATION
# --------------------------------------------------

retry_logs.clear()

few_results, few_tokens = run_classification(
    few_shot_system,
    "FEW-SHOT"
)

few_retry_count = len(retry_logs)

few_retry_logs = pd.DataFrame(retry_logs)

few_retry_logs.to_csv(
    "outputs/few_shot_retry_log.csv",
    index=False
)


# --------------------------------------------------
# 8. FINAL SUMMARY
# --------------------------------------------------

print("\n===== FINAL LLM SUMMARY =====")

print("Zero-shot tickets:", len(zero_results))
print("Few-shot tickets:", len(few_results))

print("Zero-shot tokens:", zero_tokens)
print("Few-shot tokens:", few_tokens)

print("Zero-shot retries:", zero_retry_count)
print("Few-shot retries:", few_retry_count)

print(
    "\nTotal transient errors recovered:",
    zero_retry_count + few_retry_count
)

print("\nALL LLM CLASSIFICATION CALLS COMPLETED!")



# --------------------------------------------------
# 9. TOKEN BUDGET DEMONSTRATION
# --------------------------------------------------

TOKEN_BUDGET = 6000

def run_budget_demo():

    results = []
    total_tokens = 0
    processed_count = 0
    exhausted_count = 0

    retry_logs.clear()

    # Reset immediately before this independent run
    mock_llm.reset()

    print("\n===== TOKEN BUDGET DEMO STARTED =====")
    print("Token budget:", TOKEN_BUDGET)

    for _, row in eligible_df.iterrows():

        ticket_id = row["ticket_id"]
        ticket_text = str(row["redacted_text"])

        # Check budget before starting another ticket
        if total_tokens >= TOKEN_BUDGET:

            exhausted_count += 1

            results.append({
                "ticket_id": ticket_id,
                "raw_response": "",
                "tokens_used": 0,
                "status": "budget_exhausted"
            })

            continue

        messages = [
            {
                "role": "system",
                "content": zero_shot_system
            },
            {
                "role": "user",
                "content": f"Ticket:\n{ticket_text}"
            }
        ]

        response = call_llm_with_retry(
            ticket_id,
            messages
        )

        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens

        total_tokens += tokens
        processed_count += 1

        results.append({
            "ticket_id": ticket_id,
            "raw_response": answer,
            "tokens_used": tokens,
            "status": "processed"
        })

    # Save all 156 tickets, including skipped tickets
    result_df = pd.DataFrame(results)

    result_df.to_csv(
        "outputs/triage_budget_demo.csv",
        index=False
    )

    # Save retry log for this run
    pd.DataFrame(retry_logs).to_csv(
        "outputs/budget_demo_retry_log.csv",
        index=False
    )

    print("\n===== TOKEN BUDGET RESULTS =====")
    print("Total eligible tickets:", len(eligible_df))
    print("Tickets processed:", processed_count)
    print("Tickets skipped:", exhausted_count)
    print("Tokens consumed:", total_tokens)
    print("Token budget:", TOKEN_BUDGET)
    print("Retries:", len(retry_logs))

    print("\nStatus distribution:")
    print(result_df["status"].value_counts())

    print("\nOutput: outputs/triage_budget_demo.csv")
    print("Retry log: outputs/budget_demo_retry_log.csv")

    assert processed_count + exhausted_count == len(eligible_df)
    assert exhausted_count > 0

    print("\nTOKEN BUDGET CHECKPOINT PASSED!")


run_budget_demo()
