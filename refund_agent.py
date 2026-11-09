
import csv
import json
import os
import re
import time
import uuid

import mock_llm
from mock_llm import chat
from refund_tools import TOOL_MAP


SCENARIO_FILE = "data/raw/agent_scenarios.csv"
OUTPUT_FILE = "outputs/agent_runs.jsonl"
STATE_DIR = "state"

os.makedirs("outputs", exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)


# ---------------------------------------------------------
# TOOL SPECIFICATIONS
# ---------------------------------------------------------

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Retrieve order details using an order ID.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search refund policy documents.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate refund eligibility and amount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": "number"}
                },
                "required": ["order_amount", "delivery_minutes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue an eligible automatic refund.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate a high-value refund for human approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    }
]


SYSTEM_PROMPT = """
You are an AGENT for Instamart customer support.

Use tools to investigate customer refund requests.

Rules:
1. Retrieve order information before deciding.
2. Only completed orders are eligible.
3. Search relevant policy information.
4. Calculate refund eligibility before issuing a refund.
5. Refunds above INR 1000 require human approval.
6. Never issue a zero-value refund.
7. Never invent order information or policy rules.
8. Provide a clear final answer after using tools.
"""


# ---------------------------------------------------------
# CHECKPOINT MANAGEMENT
# ---------------------------------------------------------

class SimulatedAgentCrash(Exception):
    pass


def checkpoint_path(run_id):
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", run_id)
    return os.path.join(STATE_DIR, safe_id + ".json")


def save_checkpoint(state):
    path = checkpoint_path(state["run_id"])
    temporary_path = path + ".tmp"

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, default=str)

    os.replace(temporary_path, path)


def load_checkpoint(run_id):
    path = checkpoint_path(run_id)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {run_id}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------
# TOOL EXECUTION
# ---------------------------------------------------------

def execute_tool(tool_name, arguments):
    if tool_name not in TOOL_MAP:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}

    try:
        return TOOL_MAP[tool_name](**arguments)
    except Exception as error:
        return {"success": False, "error": str(error)}


def prepare_tool_observation(tool_name, observation):
    """
    Adapt get_order's nested result to the flat structure
    expected by the supplied mock_llm.py.

    Preserve the original result for logging and checkpointing.
    """

    if (
        tool_name == "get_order"
        and observation.get("success")
        and isinstance(observation.get("order"), dict)
    ):
        return {
            "success": True,
            **observation["order"]
        }

    return observation


# ---------------------------------------------------------
# RESULT HELPERS
# ---------------------------------------------------------

def make_fallback(state, status, reason):
    return {
        "status": status,
        "final_answer": (
            f"I could not safely complete this request. {reason} "
            "Please hand this case over to a human support agent."
        ),
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "handoff_summary": {
            "scenario_id": state["scenario_id"],
            "tools_executed": state["tools_executed"],
            "reason": reason
        }
    }


def make_result(state, status, answer):
    return {
        "status": status,
        "final_answer": answer,
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "handoff_summary": None
    }


def determine_status(answer):
    if answer.startswith("REFUND:"):
        return "REFUND"
    if answer.startswith("ESCALATE:"):
        return "ESCALATE"
    if answer.startswith("DENIED:"):
        return "DENIED"
    if answer.startswith("NOT_FOUND:"):
        return "NOT_FOUND"
    if answer.startswith("NEEDS_INFO:"):
        return "NEEDS_INFO"
    return "COMPLETED"


# ---------------------------------------------------------
# REACT EXECUTION LOOP
# ---------------------------------------------------------

def continue_agent(
    state,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None
):
    start_time = time.monotonic()

    while state["step_number"] < max_iterations:

        # Time limit check before the next model call.
        if time.monotonic() - start_time >= time_limit_seconds:
            result = make_fallback(
                state,
                "FALLBACK_TIMEOUT",
                "The execution time limit was reached."
            )
            state["status"] = result["status"]
            save_checkpoint(state)
            return result

        response = chat(state["messages"], tools=TOOLS_SPEC)

        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens

        # Stop before executing tools if the token limit is exceeded.
        if state["tokens_used"] > token_limit:
            result = make_fallback(
                state,
                "FALLBACK_TOKEN_LIMIT",
                "The token budget was exceeded."
            )
            state["status"] = result["status"]
            save_checkpoint(state)
            return result

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        # Final response from the model.
        if not tool_calls:
            answer = message.content or "No final answer returned."
            status = determine_status(answer)

            state["status"] = status
            save_checkpoint(state)

            return make_result(state, status, answer)

        # Add assistant tool-call message to conversation history.
        assistant_calls = []

        for tool_call in tool_calls:
            assistant_calls.append({
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(
                        tool_call.arguments,
                        default=str
                    )
                }
            })

        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_calls
        })

        # Execute each requested tool.
        for tool_call in tool_calls:
            tool_name = tool_call.name
            arguments = tool_call.arguments

            observation = execute_tool(tool_name, arguments)

            step = {
                "step": state["step_number"] + 1,
                "tool": tool_name,
                "args": arguments,
                "observation": observation
            }

            state["steps"].append(step)
            state["tools_executed"].append(tool_name)
            state["step_number"] += 1

            # Send the adapted observation to the mock LLM.
            llm_observation = prepare_tool_observation(
                tool_name,
                observation
            )

            state["messages"].append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(llm_observation, default=str)
            })

            # Save checkpoint after every tool observation.
            save_checkpoint(state)

            if (
                crash_after_tools is not None
                and len(state["tools_executed"]) >= crash_after_tools
            ):
                raise SimulatedAgentCrash(
                    f"Simulated crash after {crash_after_tools} tools."
                )

    # Maximum iteration limit reached.
    result = make_fallback(
        state,
        "FALLBACK_MAX_ITERATIONS",
        "The maximum number of agent iterations was reached."
    )

    state["status"] = result["status"]
    save_checkpoint(state)
    return result


# ---------------------------------------------------------
# START NEW AGENT RUN
# ---------------------------------------------------------

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tools=None
):
    if run_id is None:
        run_id = f"{scenario_id}_{uuid.uuid4().hex[:8]}"

    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "tokens_used": 0,
        "llm_calls": 0,
        "step_number": 0,
        "tools_executed": [],
        "steps": [],
        "status": "RUNNING"
    }

    save_checkpoint(state)

    return continue_agent(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after_tools=crash_after_tools
    )


# ---------------------------------------------------------
# RESUME AGENT
# ---------------------------------------------------------

def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):
    state = load_checkpoint(run_id)

    if state["status"] not in ("RUNNING", "FALLBACK_TIMEOUT"):
        raise ValueError(
            f"Cannot resume run with status {state['status']}"
        )

    state["status"] = "RUNNING"

    return continue_agent(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds
    )


# ---------------------------------------------------------
# SCENARIO EXECUTION
# ---------------------------------------------------------

def load_scenarios():
    with open(
        SCENARIO_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        return list(csv.DictReader(file))


def save_scenario_result(scenario_id, result):
    record = {
        "scenario_id": scenario_id,
        "status": result["status"],
        "final_answer": result["final_answer"],
        "llm_calls": result["llm_calls"],
        "tokens_used": result["tokens_used"],
        "steps": [
            {
                "tool": step["tool"],
                "args": step["args"],
                "observation": step["observation"]
            }
            for step in result["steps"]
        ]
    }

    with open(OUTPUT_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, default=str) + "\n")


def run_all_scenarios():
    scenarios = load_scenarios()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        pass

    results = []

    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]

        mock_llm.reset()

        result = run_agent(
            user_message=scenario["message"],
            scenario_id=scenario_id
        )

        save_scenario_result(scenario_id, result)
        results.append(result)

        print("\n----------------------------------------")
        print("Scenario:", scenario_id)
        print("Expected:", scenario["expected_outcome"])
        print("Actual:", result["status"])
        print("Answer:", result["final_answer"])
        print("LLM calls:", result["llm_calls"])
        print("Tokens:", result["tokens_used"])
        print("Tools:", [s["tool"] for s in result["steps"]])

    return results


# ---------------------------------------------------------
# SAFETY TESTS
# ---------------------------------------------------------

def run_safety_tests():
    print("\n========== TOKEN LIMIT TEST ==========")

    mock_llm.reset()

    token_result = run_agent(
        "My order ORD-100048 reached me very late. I would like a refund.",
        "TOKEN_LIMIT_TEST",
        token_limit=200
    )

    print("Status:", token_result["status"])
    print("Tokens:", token_result["tokens_used"])
    print("Tools executed:", [s["tool"] for s in token_result["steps"]])

    print("\n========== TIME LIMIT TEST ==========")

    mock_llm.reset()

    timeout_result = run_agent(
        "Refund order ORD-100048.",
        "TIMEOUT_TEST",
        time_limit_seconds=0
    )

    print("Status:", timeout_result["status"])
    print("LLM calls:", timeout_result["llm_calls"])
    print("Tools executed:", [s["tool"] for s in timeout_result["steps"]])


# ---------------------------------------------------------
# CHECKPOINT AND RESUME DEMONSTRATION
# ---------------------------------------------------------

def run_resume_demo():
    print("\n========== CHECKPOINT / RESUME TEST ==========")

    run_id = "S1_RESUME_DEMO"
    path = checkpoint_path(run_id)

    if os.path.exists(path):
        os.remove(path)

    mock_llm.reset()

    try:
        run_agent(
            "My order ORD-100048 reached me very late. I would like a refund.",
            "S1_RESUME_DEMO",
            run_id=run_id,
            crash_after_tools=2
        )

    except SimulatedAgentCrash as error:
        print("Simulated crash:", error)

    checkpoint = load_checkpoint(run_id)

    print("Checkpoint saved.")
    print("Tools before resume:", checkpoint["tools_executed"])

    result = resume_agent(run_id)

    print("Resumed status:", result["status"])
    print("Final answer:", result["final_answer"])
    print("Tools after resume:", [s["tool"] for s in result["steps"]])
    print("Total LLM calls:", result["llm_calls"])

    return result


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    print("INSTAMART REACT REFUND AGENT")
    print("========================================")

    print("\nRunning all seven assignment scenarios...")
    results = run_all_scenarios()

    print("\nRunning safety tests...")
    run_safety_tests()

    print("\nRunning checkpoint and resume demonstration...")
    resume_result = run_resume_demo()

    print("\n========================================")
    print("AGENT EXECUTION SUMMARY")
    print("========================================")

    for result in results:
        print(
            result["status"],
            "|",
            result["llm_calls"],
            "LLM calls |",
            result["tokens_used"],
            "tokens"
        )

    print("\nResults saved to:", OUTPUT_FILE)
    print("Resume demonstration:", resume_result["status"])
