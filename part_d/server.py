"""Part D front-end server: serves frontend/index.html (React) and a small JSON API.

Run from the repository root:  python part_d/server.py      then open http://127.0.0.1:8000
Standard library only (http.server). Policy questions are answered by the LangChain RAG chain;
refund requests run the ReAct agent. Runs started from the web page write to their own ledger
(outputs/ui_refund_ledger.jsonl) and checkpoints (state/ui/), both git-ignored, so the graded
outputs/refund_ledger.jsonl and state/ files are never touched.
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent                                      # noqa: E402
import rag                                        # noqa: E402
from rag_langchain import answer_langchain        # noqa: E402

agent.LEDGER = os.path.join("outputs", "ui_refund_ledger.jsonl")
agent.STATE = os.path.join("state", "ui")
INDEX_HTML = os.path.join("frontend", "index.html")
HOST, PORT = "127.0.0.1", int(os.environ.get("PORT", "8000"))


def scenarios():
    return pd.read_csv(os.path.join("data", "raw", "agent_scenarios.csv")).to_dict("records")


def ask(body):
    question = str(body.get("question", "")).strip()
    if not question:
        return 400, {"error": "question is required"}
    out = answer_langchain(question)
    return 200, {"question": question, "answer": out["answer"], "sources": out["sources"],
                 "chunks": rag.retrieve(question, k=3), "engine": "LangChain RAG chain + mock_llm"}


def run(body):
    message = str(body.get("message", "")).strip()
    if not message:
        return 400, {"error": "message is required"}
    limits = {k: body[k] for k in ("max_iterations", "token_limit", "time_limit_seconds") if k in body}
    run_id = f"ui-{int(time.time() * 1000)}"
    result = agent.run_agent(message, body.get("scenario_id") or "custom", run_id=run_id,
                             **{k: (float(v) if k == "time_limit_seconds" else int(v)) for k, v in limits.items()})
    return 200, result


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload, content_type="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(INDEX_HTML, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if self.path == "/api/scenarios":
            return self._send(200, scenarios())
        if self.path in ("/vendor/react.production.min.js", "/vendor/react-dom.production.min.js"):
            with open(os.path.join("frontend", self.path.lstrip("/")), "rb") as f:   # React, shipped locally
                return self._send(200, f.read(), "text/javascript; charset=utf-8")
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        routes = {"/api/ask": ask, "/api/agent": run}
        if self.path not in routes:
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            code, payload = routes[self.path](body)
        except Exception as err:                       # report errors to the page, keep serving
            code, payload = 500, {"error": f"{type(err).__name__}: {err}"}
        self._send(code, payload)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[server] {self.address_string()} {fmt % args}\n")


if __name__ == "__main__":
    print(f"Instamart Copilot running at http://{HOST}:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
