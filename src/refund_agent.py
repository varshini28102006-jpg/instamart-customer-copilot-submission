"""refund_agent.py -- ReAct-style autonomous refund agent with safety limits, ledger logging, and crash-resumption."""

import os
import sys
import json
import time
import sqlite3
from datetime import datetime, timezone
import pandas as pd
from src.policy_rag import chunk_documents, PolicyRetriever
import mock_llm


DB_PATH = "data/instamart.db"
RUNS_LOG_PATH = "outputs/agent_runs.jsonl"
LEDGER_PATH = "outputs/refund_ledger.jsonl"
STATE_DIR = "state"


# Initialize retriever
_chunks = chunk_documents()
_retriever = PolicyRetriever(_chunks)


# ---------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------

def get_order(order_id):
    """Fetches order record from SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
        (order_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "order_id": row[0],
            "status": row[1],
            "order_amount": float(row[2]),
            "delivery_minutes": row[3] if row[3] is not None else ""
        }
    return {"error": "order not found"}


def search_policy(query):
    """Searches internal policy documents using TF-IDF retriever."""
    results = _retriever.retrieve(query, k=3)
    return {"results": results}


def calculate_refund(order_amount, delivery_minutes):
    """Calculates refund based on delivery time policy."""
    try:
        deliv = float(delivery_minutes)
    except (ValueError, TypeError):
        deliv = 0.0

    if deliv > 45:
        return {"refund_inr": float(order_amount)}
    return {"refund_inr": 0.0}


def _record_ledger_entry(action, order_id, refund_inr):
    """Appends an action to the refund ledger."""
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2)
    }
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def issue_refund(order_id, refund_inr):
    """Issues approved refund under INR 1000 and records in ledger."""
    _record_ledger_entry("issue_refund", order_id, refund_inr)
    return {"refund_inr": round(float(refund_inr), 2)}


def escalate_to_human(order_id, refund_inr):
    """Escalates high-value refund (> INR 1000) to human lead and records in ledger."""
    _record_ledger_entry("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": round(float(refund_inr), 2)}


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Reads data/instamart.db to fetch order_id, status, order_amount, and delivery_minutes.",
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
            "description": "Searches store policies using TF-IDF cosine similarity retrieval.",
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
            "description": "Computes refund amount based on delivery duration and status.",
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
            "description": "Dispatches approved refund under INR 1000 to customer and logs to ledger.",
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
            "description": "Escalates refund above INR 1000 to human support lead for authorization.",
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


# ---------------------------------------------------------------------
# Checkpointing and State Persistence
# ---------------------------------------------------------------------

def save_checkpoint(run_id, scenario_id, messages, tokens_used, step_number, executed_tools, steps):
    """Saves agent execution state to state/<run_id>.json."""
    os.makedirs(STATE_DIR, exist_ok=True)
    state_data = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_number": step_number,
        "executed_tools": executed_tools,
        "steps": steps,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    filepath = os.path.join(STATE_DIR, f"{run_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state_data, f, indent=2)


def load_checkpoint(run_id):
    """Loads agent execution checkpoint."""
    filepath = os.path.join(STATE_DIR, f"{run_id}.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint for {run_id} not found.")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# ReAct Agent Execution Loop
# ---------------------------------------------------------------------

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    simulate_crash_after_tool=None,
    run_id=None
):
    """Runs autonomous refund agent with safety limits, checkpointing, and execution logging."""
    if run_id is None:
        run_id = f"run_{scenario_id}_{int(time.time() * 1000)}"

    start_time = time.monotonic()
    tokens_used = 0
    llm_calls = 0
    steps = []
    executed_tools = []

    messages = [
        {"role": "system", "content": "AGENT. You are an autonomous refund copilot for Instamart. Use provided tools."},
        {"role": "user", "content": user_message}
    ]

    for step_num in range(1, max_iterations + 1):
        # Safety Check: Timeout
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            result = {
                "scenario_id": scenario_id,
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "Handoff to human: request timed out.",
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "steps": steps,
                "executed_tools": executed_tools
            }
            log_agent_run(result)
            return result

        # Call LLM
        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens

        # Safety Check: Token Limit
        if tokens_used > token_limit:
            result = {
                "scenario_id": scenario_id,
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "Handoff to human: token budget limit reached.",
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "steps": steps,
                "executed_tools": executed_tools
            }
            log_agent_run(result)
            return result

        msg_choice = resp.choices[0].message

        # Final answer reached (no tool calls)
        if not msg_choice.tool_calls:
            final_content = msg_choice.content or ""
            status = final_content.split(":")[0].strip() if ":" in final_content else final_content.strip()
            result = {
                "scenario_id": scenario_id,
                "status": status,
                "final_answer": final_content,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "steps": steps,
                "executed_tools": executed_tools
            }
            log_agent_run(result)
            return result

        # Tool execution
        tc = msg_choice.tool_calls[0]
        tool_name = tc.name
        tool_args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)

        # Append assistant tool call message
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc.id, "name": tool_name, "arguments": tool_args}]
        })

        # Execute mapped tool
        tool_fn = TOOL_MAP.get(tool_name)
        if not tool_fn:
            obs = {"error": f"Tool {tool_name} not recognized"}
        else:
            obs = tool_fn(**tool_args)

        executed_tools.append(tool_name)
        steps.append({
            "step": step_num,
            "tool_name": tool_name,
            "arguments": tool_args,
            "observation": obs
        })

        # Append tool observation message
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tool_name,
            "content": json.dumps(obs)
        })

        # Checkpoint after observation
        save_checkpoint(run_id, scenario_id, messages, tokens_used, step_num, executed_tools, steps)

        # Simulated crash trigger (for demonstration)
        if simulate_crash_after_tool == tool_name or simulate_crash_after_tool == len(executed_tools):
            raise RuntimeError(f"SIMULATED CRASH after tool execution: {tool_name}")

    # Fallback: Max iterations reached without terminal state
    result = {
        "scenario_id": scenario_id,
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "Handoff to human: reached maximum iterations without resolution.",
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "steps": steps,
        "executed_tools": executed_tools
    }
    log_agent_run(result)
    return result


def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):
    """Resumes agent from saved checkpoint, ensuring no duplicate tool execution."""
    checkpoint = load_checkpoint(run_id)
    scenario_id = checkpoint["scenario_id"]
    messages = checkpoint["messages"]
    tokens_used = checkpoint["tokens_used"]
    steps = checkpoint["steps"]
    executed_tools = list(checkpoint["executed_tools"])
    start_step = len(steps) + 1

    start_time = time.monotonic()
    llm_calls = 0

    for step_num in range(start_step, max_iterations + 1):
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            result = {
                "scenario_id": scenario_id,
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "Handoff to human: request timed out.",
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "steps": steps,
                "executed_tools": executed_tools
            }
            log_agent_run(result)
            return result

        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens

        if tokens_used > token_limit:
            result = {
                "scenario_id": scenario_id,
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "Handoff to human: token budget limit reached.",
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
                "steps": steps,
                "executed_tools": executed_tools
            }
            log_agent_run(result)
            return result

        msg_choice = resp.choices[0].message
        if not msg_choice.tool_calls:
            final_content = msg_choice.content or ""
            status = final_content.split(":")[0].strip() if ":" in final_content else final_content.strip()
            result = {
                "scenario_id": scenario_id,
                "status": status,
                "final_answer": final_content,
                "llm_calls": len(steps) + llm_calls,
                "tokens_used": tokens_used,
                "steps": steps,
                "executed_tools": executed_tools
            }
            log_agent_run(result)
            return result

        tc = msg_choice.tool_calls[0]
        tool_name = tc.name
        tool_args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc.id, "name": tool_name, "arguments": tool_args}]
        })

        tool_fn = TOOL_MAP[tool_name]
        obs = tool_fn(**tool_args)
        executed_tools.append(tool_name)
        steps.append({
            "step": step_num,
            "tool_name": tool_name,
            "arguments": tool_args,
            "observation": obs
        })

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tool_name,
            "content": json.dumps(obs)
        })

        save_checkpoint(run_id, scenario_id, messages, tokens_used, step_num, executed_tools, steps)

    result = {
        "scenario_id": scenario_id,
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "Handoff to human: reached maximum iterations without resolution.",
        "llm_calls": len(steps) + llm_calls,
        "tokens_used": tokens_used,
        "steps": steps,
        "executed_tools": executed_tools
    }
    log_agent_run(result)
    return result


def log_agent_run(run_result):
    """Appends agent run record to outputs/agent_runs.jsonl."""
    os.makedirs(os.path.dirname(RUNS_LOG_PATH), exist_ok=True)
    with open(RUNS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_result) + "\n")


def run_all_scenarios(scenarios_csv="data/raw/agent_scenarios.csv"):
    """Runs all 7 benchmark refund scenarios."""
    scenarios = pd.read_csv(scenarios_csv)
    # Clear runs log for fresh execution
    if os.path.exists(RUNS_LOG_PATH):
        os.remove(RUNS_LOG_PATH)
    if os.path.exists(LEDGER_PATH):
        os.remove(LEDGER_PATH)

    results = []
    for _, row in scenarios.iterrows():
        sid = row["scenario_id"]
        msg = row["message"]
        res = run_agent(msg, sid)
        results.append(res)
    return results


def demonstrate_crash_and_resume():
    """Demonstrates crash right after second tool execution and resumes seamlessly."""
    scenarios = pd.read_csv("data/raw/agent_scenarios.csv")
    s1 = scenarios.iloc[0]
    run_id = "demo_crash_s1"

    # Part 1: Run until crash after 2nd tool
    print("Initiating S1 run with simulated crash after tool 2...")
    try:
        run_agent(s1["message"], s1["scenario_id"], simulate_crash_after_tool=2, run_id=run_id)
    except RuntimeError as e:
        print(f"Caught expected crash: {e}")

    # Inspect checkpoint
    cp = load_checkpoint(run_id)
    tools_before = list(cp["executed_tools"])
    print(f"Tools executed before crash: {tools_before}")

    # Part 2: Resume
    print("Resuming agent from checkpoint...")
    final_res = resume_agent(run_id)
    tools_total = final_res["executed_tools"]
    print(f"All tools executed across run: {tools_total}")
    print(f"Final status: {final_res['status']}, Answer: {final_res['final_answer']}")

    return tools_before, tools_total, final_res


if __name__ == "__main__":
    print("--- Running 7 Scenarios ---")
    sc_results = run_all_scenarios()
    for r in sc_results:
        print(f"{r['scenario_id']}: status={r['status']}, calls={r['llm_calls']}, tokens={r['tokens_used']}, answer={r['final_answer']}")

    print("\n--- Demonstrating Crash & Resume ---")
    demonstrate_crash_and_resume()
