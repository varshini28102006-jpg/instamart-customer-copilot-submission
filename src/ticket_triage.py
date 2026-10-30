"""ticket_triage.py -- Support ticket triage pipeline with guardrails, redaction, retries, token budgeting, and routing."""

import os
import re
import json
import time
import random
import pandas as pd
import mock_llm


EMAIL_REGEX = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_REGEX = re.compile(r"\b[6-9]\d{9}\b")
ORDER_REGEX = re.compile(r"ORD-\d{6}")

ALLOWED_CATEGORIES = ["billing", "delivery", "technical", "fraud"]
BASE_DELAY = 0.05
MAX_INPUT_TOKENS = 400

ZERO_SHOT_SYSTEM_PROMPT = """CLASSIFY the support ticket into exactly one of: billing, delivery, technical, fraud.
Reply strictly in JSON: {"category": "...", "reason": "..."}."""

FEW_SHOT_SYSTEM_PROMPT = """CLASSIFY the support ticket into exactly one of: billing, delivery, technical, fraud.
Reply strictly in JSON: {"category": "...", "reason": "..."}.
EXAMPLES:
Ticket: I was charged twice for order [ORDER_ID].
Output: {"category": "billing", "reason": "duplicate charge"}
Ticket: There is an unauthorized transaction on my card for order [ORDER_ID].
Output: {"category": "fraud", "reason": "unauthorized card transaction"}"""


def redact_pii(text):
    """Redacts email, phone, and order IDs in strict sequence."""
    if not text:
        return text, 0, 0, 0

    e_count = len(EMAIL_REGEX.findall(text))
    t1 = EMAIL_REGEX.sub("[EMAIL]", text)

    p_count = len(PHONE_REGEX.findall(t1))
    t2 = PHONE_REGEX.sub("[PHONE]", t1)

    o_count = len(ORDER_REGEX.findall(t2))
    redacted = ORDER_REGEX.sub("[ORDER_ID]", t2)

    return redacted, e_count, p_count, o_count


def verify_no_pii(text):
    """Verifies that no unredacted PII remains."""
    return (
        len(EMAIL_REGEX.findall(text)) == 0 and
        len(PHONE_REGEX.findall(text)) == 0 and
        len(ORDER_REGEX.findall(text)) == 0
    )


def call_llm_with_retry(messages, ticket_id, max_retries=3, base_delay=BASE_DELAY, verbose=False):
    """Wraps model calls in retry with exponential backoff and jitter."""
    for n in range(max_retries + 1):
        try:
            resp = mock_llm.chat(messages)
            return resp
        except mock_llm.TransientLLMError as e:
            if n < max_retries:
                wait_sec = base_delay * (2 ** n) * random.uniform(0.9, 1.15)
                if verbose:
                    print(f"Transient error on {ticket_id} (retry {n}, waiting {wait_sec:.4f}s): {e}")
                time.sleep(wait_sec)
            else:
                raise e


def parse_and_validate_response(content):
    """Validates model response: JSON check, word-level fallback, or flagged invalid."""
    if not content:
        return None, "flagged_invalid_output"

    # 1. JSON parsing check
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            cat = data.get("category")
            if cat in ALLOWED_CATEGORIES:
                return cat, "classified_json"
            else:
                return None, "flagged_invalid_output"
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Keyword fallback via re.findall(r"[a-z]+")
    words = re.findall(r"[a-z]+", content.lower())
    found_cats = [w for w in words if w in ALLOWED_CATEGORIES]
    if len(found_cats) == 1:
        return found_cats[0], "classified_keyword"

    return None, "flagged_invalid_output"


def run_triage_pipeline(
    tickets_path="data/raw/tickets.csv",
    mode="few_shot",
    token_budget=None,
    segments_path="outputs/customer_segments.csv",
    verbose=False
):
    """Executes the triage pipeline on support tickets in file order."""
    tickets = pd.read_csv(tickets_path, keep_default_na=False)

    # Load segment information for routing
    segments_map = {}
    if os.path.exists(segments_path):
        seg_df = pd.read_csv(segments_path)
        for _, r in seg_df.iterrows():
            segments_map[r["customer_id"]] = {
                "segment_name": r["segment_name"],
                "is_premium": bool(r["is_premium"])
            }

    system_prompt = FEW_SHOT_SYSTEM_PROMPT if mode == "few_shot" else ZERO_SHOT_SYSTEM_PROMPT
    mock_llm.reset()
    random.seed(42)

    total_tokens_used = 0
    results = []
    transient_errors_logged = 0

    for _, row in tickets.iterrows():
        tid = row["ticket_id"]
        cid = row["customer_id"]
        raw_text = row["text"]
        true_cat = row.get("true_category", "")

        # Guardrail 1: Empty ticket
        if not raw_text or not raw_text.strip():
            results.append({
                "ticket_id": tid,
                "customer_id": cid,
                "status": "skipped_empty",
                "category": "",
                "true_category": true_cat,
                "segment_name": segments_map.get(cid, {}).get("segment_name", ""),
                "priority": "NORMAL",
                "route": "human_review",
                "tokens_used": 0
            })
            continue

        # Guardrail 2: Excessively long ticket
        est_tok = mock_llm.est_tokens(raw_text)
        if est_tok > MAX_INPUT_TOKENS:
            results.append({
                "ticket_id": tid,
                "customer_id": cid,
                "status": "flagged_long",
                "category": "",
                "true_category": true_cat,
                "segment_name": segments_map.get(cid, {}).get("segment_name", ""),
                "priority": "NORMAL",
                "route": "human_review",
                "tokens_used": 0
            })
            continue

        # Check Token Budget before model call
        if token_budget is not None and total_tokens_used >= token_budget:
            results.append({
                "ticket_id": tid,
                "customer_id": cid,
                "status": "budget_exhausted",
                "category": "",
                "true_category": true_cat,
                "segment_name": segments_map.get(cid, {}).get("segment_name", ""),
                "priority": "NORMAL",
                "route": "human_review",
                "tokens_used": 0
            })
            continue

        # PII Redaction
        redacted_text, _, _, _ = redact_pii(raw_text)
        assert verify_no_pii(redacted_text), f"Unredacted PII found in {tid}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": redacted_text}
        ]

        # Call with retry
        try:
            resp = call_llm_with_retry(messages, tid, max_retries=3, verbose=verbose)
            call_tokens = resp.usage.total_tokens
            total_tokens_used += call_tokens
            content = resp.choices[0].message.content
        except mock_llm.TransientLLMError:
            transient_errors_logged += 1
            results.append({
                "ticket_id": tid,
                "customer_id": cid,
                "status": "flagged_invalid_output",
                "category": "",
                "true_category": true_cat,
                "segment_name": segments_map.get(cid, {}).get("segment_name", ""),
                "priority": "NORMAL",
                "route": "human_review",
                "tokens_used": 0
            })
            continue

        cat, status = parse_and_validate_response(content)

        # Routing logic
        is_premium = segments_map.get(cid, {}).get("is_premium", False)
        seg_name = segments_map.get(cid, {}).get("segment_name", "")

        if status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
            route = "human_review"
            priority = "NORMAL"
        elif cat == "fraud":
            route = "security_team"
            priority = "HIGH"
        elif is_premium:
            route = "premium_desk"
            priority = "HIGH"
        else:
            route = "support_queue"
            priority = "NORMAL"

        results.append({
            "ticket_id": tid,
            "customer_id": cid,
            "status": status,
            "category": cat if cat else "",
            "true_category": true_cat,
            "segment_name": seg_name,
            "priority": priority,
            "route": route,
            "tokens_used": call_tokens
        })

    res_df = pd.DataFrame(results)
    return res_df, total_tokens_used


def generate_outputs():
    """Generates triage_results.csv and triage_budget_demo.csv."""
    os.makedirs("outputs", exist_ok=True)

    # 1. Budget Demonstration (Zero-Shot, budget=6000)
    df_budget, tokens_budget = run_triage_pipeline(mode="zero_shot", token_budget=6000)
    out_cols = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
    df_budget[out_cols].to_csv("outputs/triage_budget_demo.csv", index=False)

    # 2. Final Triage Results (Few-Shot, budget=100000)
    df_final, tokens_final = run_triage_pipeline(mode="few_shot", token_budget=100000)
    df_final[out_cols].to_csv("outputs/triage_results.csv", index=False)

    return df_budget, df_final


if __name__ == "__main__":
    b_df, f_df = generate_outputs()
    print("--- Budget Demo ---")
    print(b_df["status"].value_counts())
    print("\n--- Final Few-Shot Triage ---")
    print(f_df["status"].value_counts())
    print("\nFinal Routing:")
    print(f_df["route"].value_counts())
