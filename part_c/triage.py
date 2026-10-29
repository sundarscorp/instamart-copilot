"""Part C -- support-ticket triage with guardrails, around the offline mock_llm.py.

Run from the repository root:  python part_c/triage.py
Tickets are processed in file order. Nothing personal (email, phone, order id) and nothing
empty or over-long ever reaches the model.
"""
import json
import os
import random
import re
import sys
import time
from collections import Counter
import pandas as pd

sys.path.insert(0, os.path.abspath("."))          # mock_llm.py lives at the repository root
import mock_llm                                    # noqa: E402  (supplied file, used unmodified)

RAW, OUT = os.path.join("data", "raw"), "outputs"
MAX_INPUT_TOKENS = 400
DEMO_BUDGET, FINAL_BUDGET = 6000, 100000
OUTPUT_COLUMNS = ["ticket_id", "customer_id", "status", "category", "segment_name",
                  "priority", "route", "tokens_used"]
HUMAN_REVIEW = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}

# Applied IN THIS ORDER: emails first, so digits inside an address are never half-masked
# by the phone or order-id patterns.
REDACTIONS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]


CATEGORIES = ["billing", "delivery", "technical", "fraud"]
MAX_RETRIES, BASE_DELAY = 3, 0.05
jitter = random.Random(42)                         # seeded so the retry waits are reproducible

# ---------------- Task 3: two prompts ----------------
ZERO_SHOT = """You CLASSIFY customer support tickets for a grocery delivery app.
Categories: billing, delivery, technical, fraud
Reply with JSON only: {"category": "...", "reason": "..."}"""

FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: I was charged on my saved card for [ORDER_ID] that I never authorized. -> {"category": "fraud", "reason": "unauthorized card use"}
Ticket: The app crashes on the payment screen every time I pay by UPI. -> {"category": "technical", "reason": "app crash at checkout"}
Ticket: Order [ORDER_ID] arrived without the milk and bread I ordered. -> {"category": "delivery", "reason": "items missing from delivery"}
Ticket: I was debited twice for order [ORDER_ID] but placed it once. -> {"category": "billing", "reason": "duplicate charge"}"""


def load_tickets():
    # keep_default_na=False: empty ticket texts stay "" instead of becoming NaN
    return pd.read_csv(os.path.join(RAW, "tickets.csv"), keep_default_na=False)


def input_guardrail(text):
    """Return a status if the ticket must NOT be sent to the model, else None."""
    if not text.strip():
        return "skipped_empty"
    if mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"
    return None


def redact(text):
    """Mask personal details; return (redacted_text, {type: substitutions})."""
    counts = {}
    for name, pattern, placeholder in REDACTIONS:
        text, n = pattern.subn(placeholder, text)
        counts[name] = n
    return text, counts


def contains_pii(text):
    return any(pattern.search(text) for _, pattern, _ in REDACTIONS)


# ---------------- Task 4: resilient calls ----------------
def call_with_retry(messages, ticket_id, stats):
    """Call the model; on TransientLLMError wait BASE_DELAY * 2**n * U(0.9, 1.15) and retry (max 3)."""
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError as err:
            stats["transient_errors"] += 1
            if n == MAX_RETRIES:
                raise                                  # permanent failure after 3 retries
            wait = BASE_DELAY * 2 ** n * jitter.uniform(0.9, 1.15)
            stats["waits"].append((n, wait))
            print(f"  retry  ticket={ticket_id}  retry_number={n}  wait={wait:.4f}s  ({err})")
            time.sleep(wait)


# ---------------- Task 5: validate every answer ----------------
def validate(reply):
    """Return (status, category). Never guesses: anything doubtful goes to human review."""
    try:
        data = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (reply or "").lower()))   # 'billing-related' -> billing, related
        found = [c for c in CATEGORIES if c in words]
        if len(found) == 1:
            return "classified_keyword", found[0]
        return "flagged_invalid_output", ""
    if isinstance(data, dict) and data.get("category") in CATEGORIES:
        return "classified_json", data["category"]
    return "flagged_invalid_output", ""               # valid JSON, category outside the list


def route(results):
    """Task 7 routing. Reads the Part B segments; a customer with no segment row is not premium."""
    seg = pd.read_csv(os.path.join(OUT, "customer_segments.csv")).set_index("customer_id")
    r = results.copy()
    r["segment_name"] = r["customer_id"].map(seg["segment_name"]).fillna("")
    premium = r["customer_id"].map(seg["is_premium"]).fillna(False).astype(bool)

    def decide(row, is_premium):
        if row["status"] in HUMAN_REVIEW:
            return "human_review", "NORMAL"
        if row["category"] == "fraud":
            return "security_team", "HIGH"
        if is_premium:
            return "premium_desk", "HIGH"
        return "support_queue", "NORMAL"

    decided = [decide(row, p) for (_, row), p in zip(r.iterrows(), premium)]
    r["route"] = [d[0] for d in decided]
    r["priority"] = [d[1] for d in decided]
    return r[OUTPUT_COLUMNS]


def run_pipeline(tickets, system_prompt, token_budget=None):
    """Triage every ticket in file order; return one result row per ticket plus run stats."""
    mock_llm.reset()                                    # independent batch run
    stats = {"transient_errors": 0, "waits": [], "tokens": 0, "calls": 0}
    rows = []
    for t in tickets.itertuples(index=False):
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "status": "",
               "category": "", "tokens_used": 0}
        status = input_guardrail(t.text)
        if status:
            row["status"] = status
        elif token_budget is not None and stats["tokens"] >= token_budget:
            row["status"] = "budget_exhausted"         # Task 7: no call once the budget is spent
        else:
            user_content, _ = redact(t.text)
            assert not contains_pii(user_content), f"PII would leak in {t.ticket_id}"
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}]   # exactly the redacted text
            resp = call_with_retry(messages, t.ticket_id, stats)
            stats["calls"] += 1
            row["tokens_used"] = resp.usage.total_tokens
            stats["tokens"] += resp.usage.total_tokens
            row["status"], row["category"] = validate(resp.choices[0].message.content)
        rows.append(row)
    return pd.DataFrame(rows), stats


def main():
    tickets = load_tickets()

    # ---------------- Task 1: input guardrails ----------------
    print("## Task 1: input guardrails\n")
    guard = tickets["text"].map(input_guardrail)
    long_tokens = sorted({mock_llm.est_tokens(t) for t in tickets.loc[guard == "flagged_long", "text"]})
    print(f"tickets: {len(tickets)}")
    print(f"skipped_empty: {(guard == 'skipped_empty').sum()}")
    print(f"flagged_long (> {MAX_INPUT_TOKENS} estimated tokens): {(guard == 'flagged_long').sum()}, "
          f"estimated at {long_tokens} tokens")
    print(f"eligible for the model: {guard.isna().sum()}")
    assert (guard == "skipped_empty").sum() == 4 and (guard == "flagged_long").sum() == 4
    assert long_tokens == [734] and guard.isna().sum() == 156

    # ---------------- Task 2: redact personal information ----------------
    print("\n## Task 2: redaction (all 164 tickets)\n")
    totals = {name: 0 for name, _, _ in REDACTIONS}
    redacted = []
    for text in tickets["text"]:
        clean, counts = redact(text)
        redacted.append(clean)
        for k, v in counts.items():
            totals[k] += v
    print("| type | substitutions |\n|---|---:|")
    for k, v in totals.items():
        print(f"| {k} | {v} |")
    print(f"| total | {sum(totals.values())} |")
    leftovers = sum(len(p.findall(t)) for t in redacted for _, p, _ in REDACTIONS)
    print(f"\nre-applying the three patterns to the redacted texts finds: {leftovers}")
    assert totals == {"email": 55, "phone": 91, "order_id": 132} and leftovers == 0
    print("\nexample:")
    print("  before:", tickets["text"].iloc[0])
    print("  after: ", redacted[0])
    print("\nTASKS 1-2 MATCH REFERENCE")

    # ---------------- Task 3: prompts ----------------
    print("\n## Task 3: prompts\n")
    for name, p in [("zero-shot", ZERO_SHOT), ("few-shot", FEW_SHOT)]:
        print(f"{name}: {mock_llm.est_tokens(p)} estimated tokens; contains CLASSIFY: {'CLASSIFY' in p}; "
              f"contains EXAMPLES: {'EXAMPLES:' in p}")
    assert "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT
    assert all(c in p for p in (ZERO_SHOT, FEW_SHOT) for c in CATEGORIES)

    # ---------------- Tasks 4-5: one full zero-shot run ----------------
    print("\n## Tasks 4-5: full zero-shot run (retry log below)\n")
    result, stats = run_pipeline(tickets, ZERO_SHOT)
    waits = [w for _, w in stats["waits"]]
    print(f"\ntransient errors: {stats['transient_errors']}, all recovered: "
          f"{len(waits) == stats['transient_errors']}; retry waits {min(waits):.4f}-{max(waits):.4f}s "
          f"(allowed for retry 0: 0.0450-0.0575s)")
    counts = Counter(result["status"])
    print("status counts:", dict(counts))
    assert stats["transient_errors"] == 19 and len(waits) == 19
    assert all(n == 0 and 0.045 <= w <= 0.0575 for n, w in stats["waits"])
    assert (counts["classified_json"], counts["classified_keyword"], counts["flagged_invalid_output"]) == (135, 12, 9)
    print("TASKS 1-5 MATCH REFERENCE")

    # ---------------- Task 6: zero-shot vs few-shot ----------------
    print("\n## Task 6: zero-shot vs few-shot (no token budget)\n")
    truth = tickets.set_index("ticket_id")["true_category"]
    runs = {"zero-shot": result, "few-shot": run_pipeline(tickets, FEW_SHOT)[0]}
    print("| prompt | classified | correct | accuracy | total tokens | tokens per classified ticket |")
    print("|---|---:|---:|---:|---:|---:|")
    acc = {}
    for name, r in runs.items():
        done = r[r["status"].isin(["classified_json", "classified_keyword"])]
        correct = (done["category"].values == truth.loc[done["ticket_id"]].values).sum()
        acc[name] = correct / len(done)
        total = r["tokens_used"].sum()
        print(f"| {name} | {len(done)} | {correct} | {acc[name]:.1%} | {total:,} | {total / len(done):.1f} |")
    assert round(acc["zero-shot"], 3) == 0.857 and acc["few-shot"] == 1.0

    zs = runs["zero-shot"]
    zs_done = zs[zs["status"].isin(["classified_json", "classified_keyword"])]
    confusion = pd.crosstab(truth.loc[zs_done["ticket_id"]].values, zs_done["category"].values,
                            rownames=["true"], colnames=["predicted"])
    print("\nzero-shot confusion table (rows = true category, columns = predicted)\n")
    print(confusion.to_string())

    # ---------------- Task 7: budget, routing, outputs ----------------
    print("\n## Task 7: token budget and routing\n")
    os.makedirs(OUT, exist_ok=True)
    demo, demo_stats = run_pipeline(tickets, ZERO_SHOT, token_budget=DEMO_BUDGET)
    n_exhausted = int((demo["status"] == "budget_exhausted").sum())
    route(demo).to_csv(os.path.join(OUT, "triage_budget_demo.csv"), index=False)
    print(f"budget demo: zero-shot, TOKEN_BUDGET = {DEMO_BUDGET:,}: {demo_stats['tokens']:,} tokens used, "
          f"{n_exhausted} tickets budget_exhausted -> outputs/triage_budget_demo.csv")
    assert n_exhausted >= 1

    final, final_stats = run_pipeline(tickets, FEW_SHOT, token_budget=FINAL_BUDGET)
    final = route(final)
    final.to_csv(os.path.join(OUT, "triage_results.csv"), index=False)
    print(f"final run: few-shot, TOKEN_BUDGET = {FINAL_BUDGET:,}: {final_stats['tokens']:,} tokens used "
          f"-> outputs/triage_results.csv ({len(final)} rows)\n")
    print(pd.crosstab(final["route"], final["priority"]).to_string())
    print("\nstatus counts:", dict(Counter(final["status"])))
    assert len(final) == 164 and list(final.columns) == OUTPUT_COLUMNS
    print("\nPART C COMPLETE")


if __name__ == "__main__":
    main()
