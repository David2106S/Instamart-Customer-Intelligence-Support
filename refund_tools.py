
import sqlite3
import policy_rag
import os
import json

from datetime import datetime, timezone


# ==================================================
# CONFIGURATION
# ==================================================

DATABASE_PATH = "data/instamart.db"
REFUND_LEDGER = "outputs/refund_ledger.jsonl"


# ==================================================
# TOOL 1: GET ORDER DETAILS
# ==================================================

def get_order(order_id):
    """Retrieve order information from the Instamart SQLite database."""

    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                order_id,
                customer_id,
                order_amount,
                status,
                delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        )

        order = cursor.fetchone()

        if order is None:
            return {
                "success": False,
                "error": f"Order {order_id} not found"
            }

        return {
            "success": True,
            "order": dict(order)
        }

    finally:
        conn.close()


# ==================================================
# TOOL 2: SEARCH POLICY
# ==================================================

def search_policy(query):
    """Retrieve the top three relevant policy chunks."""

    try:
        results = policy_rag.retrieve(query, k=3)

        return {
            "success": True,
            "results": results
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error)
        }


# ==================================================
# TOOL 3: CALCULATE REFUND
# ==================================================

def calculate_refund(order_amount, delivery_minutes):
    """
    Refund rules:
    - Delivery above 45 minutes: full refund.
    - Delivery at or below 45 minutes: no refund.
    - Refund up to INR 1000: automatic approval.
    - Refund above INR 1000: human approval.
    """

    if order_amount is None or delivery_minutes is None:
        return {
            "success": False,
            "eligible": False,
            "refund_inr": 0,
            "approval": "NOT_ELIGIBLE",
            "reason": "Order amount or delivery time is unavailable."
        }

    if delivery_minutes > 45:

        refund_amount = round(float(order_amount), 2)

        approval = (
            "AUTOMATIC"
            if refund_amount <= 1000
            else "HUMAN_APPROVAL"
        )

        return {
            "success": True,
            "eligible": True,
            "refund_inr": refund_amount,
            "approval": approval,
            "reason": "Completed order delivered after 45 minutes."
        }

    return {
        "success": True,
        "eligible": False,
        "refund_inr": 0,
        "approval": "NOT_ELIGIBLE",
        "reason": "Delivery was completed within 45 minutes."
    }


# ==================================================
# TOOL 4: ISSUE REFUND
# ==================================================

def issue_refund(order_id, refund_inr):
    """Record an approved refund in the JSONL ledger."""

    refund_inr = round(float(refund_inr), 2)

    if refund_inr <= 0:
        return {
            "success": False,
            "refund_inr": 0,
            "error": "Refund amount must be greater than zero."
        }

    os.makedirs("outputs", exist_ok=True)

    record = {
        "order_id": order_id,
        "refund_inr": refund_inr,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": "issue_refund"
    }

    with open(REFUND_LEDGER, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return {
        "success": True,
        "refund_inr": refund_inr,
        "order_id": order_id,
        "message": f"Refund of INR {refund_inr:.2f} issued for {order_id}.",
        "record": record
    }


# ==================================================
# TOOL 5: ESCALATE TO HUMAN
# ==================================================

def escalate_to_human(order_id, refund_inr):
    """Record a refund requiring human support approval."""

    refund_inr = round(float(refund_inr), 2)

    if refund_inr <= 0:
        return {
            "success": False,
            "refund_inr": 0,
            "error": "Escalation amount must be greater than zero."
        }

    os.makedirs("outputs", exist_ok=True)

    record = {
        "order_id": order_id,
        "refund_inr": refund_inr,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": "escalate_to_human"
    }

    with open(REFUND_LEDGER, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return {
        "success": True,
        "refund_inr": refund_inr,
        "order_id": order_id,
        "message": (
            f"Refund of INR {refund_inr:.2f} "
            f"for {order_id} escalated to human support."
        ),
        "record": record
    }


# ==================================================
# TOOL MAP
# ==================================================

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


# ==================================================
# TESTING ALL FIVE TOOLS
# ==================================================

if __name__ == "__main__":

    print("=" * 70)
    print("INSTAMART REFUND AGENT - TOOL TESTING")
    print("=" * 70)

    print("\nTEST 1: Existing completed order")
    print(get_order("ORD-110093"))

    print("\nTEST 2: Cancelled order")
    print(get_order("ORD-102006"))

    print("\nTEST 3: Non-existing order")
    print(get_order("ORD-999999"))

    print("\nTEST 4: Search refund policy")

    policy_result = search_policy("refund policy late delivery")

    print("Retrieval successful:", policy_result["success"])
    print("Retrieved chunks:", len(policy_result.get("results", [])))

    for result in policy_result.get("results", []):
        print("\nSource:", result["source"])
        print("Score:", round(result["score"], 4))
        print("Text:", result["text"])

    print("\nTEST 5: Eligible refund below INR 1000")
    print(calculate_refund(128.25, 60))

    print("\nTEST 6: Eligible refund above INR 1000")
    print(calculate_refund(1999.00, 70))

    print("\nTEST 7: Delivery within 45 minutes")
    print(calculate_refund(500.00, 40))

    print("\nTEST 8: Missing delivery time")
    print(calculate_refund(500.00, None))

    print("\nTEST 9: Issue refund")
    print(issue_refund("ORD-TEST-001", 128.25))

    print("\nTEST 10: Escalate to human")
    print(escalate_to_human("ORD-TEST-002", 1999.00))

    print("\nTEST 11: Reject zero refund")
    print(issue_refund("ORD-TEST-003", 0))

    print("\n" + "=" * 70)
    print("ALL TOOL TESTS COMPLETED")
    print("=" * 70)
