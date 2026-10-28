"""Part C -- support-ticket triage with guardrails, around the offline mock_llm.py.

Run from the repository root:  python part_c/triage.py
Tickets are processed in file order. Nothing personal (email, phone, order id) and nothing
empty or over-long ever reaches the model.
"""
import os
import re
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath("."))          # mock_llm.py lives at the repository root
import mock_llm                                    # noqa: E402  (supplied file, used unmodified)

RAW = os.path.join("data", "raw")
MAX_INPUT_TOKENS = 400

# Applied IN THIS ORDER: emails first, so digits inside an address are never half-masked
# by the phone or order-id patterns.
REDACTIONS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]


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


if __name__ == "__main__":
    main()
