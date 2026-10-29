"""Part D, Tasks 5-8 -- a ReAct refund agent with tools, safety limits and crash-and-resume.

Run from the repository root:  python part_d/agent.py
The agent reads orders from data/instamart.db (Part A) and searches policies with the
retrieval from part_d/rag.py. The "LLM" is mock_llm.py (AGENT behaviour): it asks for tools,
our code runs them. Writes outputs/agent_runs.jsonl, outputs/refund_ledger.jsonl and state/.

Refund rules: only completed orders; delivered in more than 45 minutes -> 100% of order_amount;
45 minutes or less -> 0; a refund above INR 1000 is escalated to a human, never issued.
"""
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mock_llm                                   # noqa: E402  (supplied file, used unmodified)
import rag                                        # noqa: E402

DB = os.path.join("data", "instamart.db")
OUT, STATE = "outputs", "state"
RUNS_FILE = os.path.join(OUT, "agent_runs.jsonl")
LEDGER = os.path.join(OUT, "refund_ledger.jsonl")       # the web server points this elsewhere
SYSTEM_PROMPT = ("You are a refund AGENT for Instamart support. Use the tools to look up the order, "
                 "check the refund policy, calculate the refund, then issue it or escalate it. "
                 "Finish with one line: REFUND, ESCALATE, DENIED or NOT_FOUND, a colon, and a short reason.")


# ---------------- Task 5: tools ----------------
def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders "
                          "WHERE order_id = ?", (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    return dict(zip(["order_id", "status", "order_amount", "delivery_minutes"], row))


def search_policy(query):
    return {"results": rag.retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    late = delivery_minutes is not None and delivery_minutes > 45
    return {"refund_inr": round(float(order_amount), 2) if late else 0.0}


def _ledger(tool, order_id, refund_inr):
    os.makedirs(os.path.dirname(LEDGER) or ".", exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "tool": tool, "order_id": order_id, "refund_inr": refund_inr}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


def _spec(name, description, **params):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": params, "required": list(params)}}


TOOLS_SPEC = [
    _spec("get_order", "Look up an order in the Instamart database by its id.",
          order_id={"type": "string", "description": "order id such as ORD-100048"}),
    _spec("search_policy", "Search the policy documents and return the three most relevant passages.",
          query={"type": "string"}),
    _spec("calculate_refund", "Apply the refund rules: full amount if delivered in more than 45 minutes, else 0.",
          order_amount={"type": "number"}, delivery_minutes={"type": ["number", "null"]}),
    _spec("issue_refund", "Issue a refund of INR 1000 or less and record it in the refund ledger.",
          order_id={"type": "string"}, refund_inr={"type": "number"}),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead and record it in the ledger.",
          order_id={"type": "string"}, refund_inr={"type": "number"}),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# ---------------- Tasks 6-8: the loop, safety limits, checkpoints ----------------
class SimulatedCrash(Exception):
    """Raised on purpose in the crash-and-resume demonstration."""


def _checkpoint(state):
    os.makedirs(STATE, exist_ok=True)
    with open(os.path.join(STATE, f"{state['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _fallback(state, status, reason):
    done = ", ".join(s["tool"] for s in state["steps"]) or "none"
    summary = (f"HANDOFF to a human agent ({reason}). Scenario {state['scenario_id']}: "
               f"{state['llm_calls']} model calls, {state['tokens_used']} tokens; tools completed: {done}.")
    return _finish(state, status, summary)


def _finish(state, status, final_answer):
    state["status"], state["final_answer"] = status, final_answer
    _checkpoint(state)
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": status,
            "final_answer": final_answer, "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"], "tools_executed": state["tools_executed"],
            "refund_inr": state.get("refund_inr", 0.0)}


def _loop(state, crash_after_tool=None):
    lim, start = state["limits"], time.monotonic()
    while True:
        if time.monotonic() - start >= lim["time_limit_seconds"]:            # (c) wall clock
            return _fallback(state, "FALLBACK_TIMEOUT", "time limit reached")
        if state["llm_calls"] >= lim["max_iterations"]:                       # (a) iterations
            return _fallback(state, "FALLBACK_MAX_ITERATIONS", f"no answer after {lim['max_iterations']} model calls")
        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > lim["token_limit"]:                         # (b) tokens: stop before any tool
            return _fallback(state, "FALLBACK_TOKEN_LIMIT", f"token limit {lim['token_limit']} exceeded")
        msg = resp.choices[0].message
        if not msg.tool_calls:                                                # final answer
            return _finish(state, msg.content.split(":", 1)[0].strip(), msg.content)
        for tc in msg.tool_calls:                                             # act, observe
            state["messages"].append({"role": "assistant", "content": None, "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
            observation = TOOL_MAP[tc.name](**tc.arguments)
            state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                      "content": json.dumps(observation)})
            state["step"] += 1
            state["tools_executed"].append(tc.name)
            state["steps"].append({"step": state["step"], "tool": tc.name, "arguments": tc.arguments,
                                   "observation": observation})
            if tc.name in ("issue_refund", "escalate_to_human"):
                state["refund_inr"] = observation["refund_inr"]
            _checkpoint(state)                                                # Task 8: after every observation
            if crash_after_tool is not None and state["step"] == crash_after_tool:
                raise SimulatedCrash(f"simulated crash after tool #{state['step']} ({tc.name})")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tool=None):
    state = {"run_id": run_id or scenario_id, "scenario_id": scenario_id,
             "limits": {"max_iterations": max_iterations, "token_limit": token_limit,
                        "time_limit_seconds": time_limit_seconds},
             "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "step": 0, "tools_executed": [], "steps": [],
             "refund_inr": 0.0, "status": None, "final_answer": None}
    return _loop(state, crash_after_tool)


def resume_agent(run_id):
    """Load the checkpoint for run_id and continue the loop from where it stopped."""
    with open(os.path.join(STATE, f"{run_id}.json"), encoding="utf-8") as f:
        state = json.load(f)
    return _loop(state)


def log_run(result, label):
    with open(RUNS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"label": label, **{k: result[k] for k in (
            "run_id", "scenario_id", "status", "final_answer", "llm_calls", "tokens_used",
            "refund_inr", "steps")}}) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    for path in (RUNS_FILE, LEDGER):                    # fresh files on every full run
        open(path, "w").close()

    print("## Task 5: tools\n")
    print(", ".join(s["name"] for s in TOOLS_SPEC), "| TOOL_MAP keys match:", list(TOOL_MAP) == [s["name"] for s in TOOLS_SPEC])

    print("\n## Task 6: seven scenarios (max_iterations=6, token_limit=6000, time_limit_seconds=10)\n")
    sc = pd.read_csv(os.path.join("data", "raw", "agent_scenarios.csv"))
    print("| scenario | expected | status | refund (INR) | model calls | tokens | tools |\n|---|---|---|---:|---:|---:|---|")
    for s in sc.itertuples(index=False):
        r = run_agent(s.message, s.scenario_id)
        log_run(r, "scenario")
        print(f"| {s.scenario_id} | {s.expected_outcome} | {r['status']} | {r['refund_inr']:,.2f} | "
              f"{r['llm_calls']} | {r['tokens_used']:,} | {' > '.join(r['tools_executed'])} |")
        ok = r["status"] == ("FALLBACK_MAX_ITERATIONS" if s.expected_outcome == "FALLBACK" else s.expected_outcome)
        assert ok and abs(r["refund_inr"] - s.expected_refund_inr) <= 0.01, (s, r["status"], r["refund_inr"])
        if s.scenario_id == "S1":
            assert r["llm_calls"] == 5
        if s.scenario_id == "S7":
            assert r["llm_calls"] == 6 and r["tokens_used"] <= 6000

    print("\n## Task 7: stop safely\n")
    s1 = sc.set_index("scenario_id").loc["S1", "message"]
    tok = run_agent(s1, "S1", token_limit=200, run_id="S1-token-limit-demo")
    tim = run_agent(s1, "S1", time_limit_seconds=0, run_id="S1-timeout-demo")
    for label, r in [("S1 with token_limit=200", tok), ("S1 with time_limit_seconds=0", tim)]:
        log_run(r, label)
        print(f"{label}: {r['status']} after {r['llm_calls']} model calls, tools {r['tools_executed']}\n"
              f"  final_answer: {r['final_answer']}")
    # the call that crossed the limit asked for calculate_refund; it must NOT have been executed
    assert tok["status"] == "FALLBACK_TOKEN_LIMIT" and tok["tokens_used"] > 200
    assert tok["tools_executed"] == ["get_order", "search_policy"]
    assert tim["status"] == "FALLBACK_TIMEOUT" and tim["llm_calls"] == 0

    print("\n## Task 8: crash and resume (S1)\n")
    try:
        run_agent(s1, "S1", run_id="S1-crash-resume", crash_after_tool=2)
    except SimulatedCrash as err:
        print(f"crashed: {err}")
    with open(os.path.join(STATE, "S1-crash-resume.json"), encoding="utf-8") as f:
        before = json.load(f)["tools_executed"]
    print(f"checkpoint state/S1-crash-resume.json holds tools executed so far: {before}")
    r = resume_agent("S1-crash-resume")
    log_run(r, "S1 crash and resume")
    print(f"resumed: {r['status']} INR {r['refund_inr']:.2f}; tools across crash and resume: {r['tools_executed']}")
    assert r["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert r["status"] == "REFUND" and abs(r["refund_inr"] - 128.25) <= 0.01

    with open(LEDGER, encoding="utf-8") as f:
        ledger = [json.loads(line) for line in f]
    print(f"\nrefund ledger: {len(ledger)} entries -> " + ", ".join(f"{e['tool']} {e['order_id']} {e['refund_inr']}" for e in ledger))
    print(f"agent runs logged: {sum(1 for _ in open(RUNS_FILE))} -> {RUNS_FILE}; checkpoints in {STATE}/: {sorted(os.listdir(STATE))}")
    print("\nPART D AGENT MATCHES REFERENCE")


if __name__ == "__main__":
    main()
