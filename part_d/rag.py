"""Part D, Tasks 1-4 -- retrieval-grounded policy assistant over data/policies/*.txt.

Run from the repository root:  python part_d/rag.py
Chunk (40 words, 10 overlap) -> TF-IDF vectors -> cosine-similarity retrieval -> grounded
answer from mock_llm.py ("POLICY ASSISTANT" behaviour). Imported by the refund agent
(part_d/agent.py) and by the LangChain version (part_d/rag_langchain.py).
"""
import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.abspath("."))
import mock_llm                                   # noqa: E402  (supplied file, used unmodified)

POLICIES = os.path.join("data", "policies")
CHUNK_WORDS, OVERLAP = 40, 10
FALLBACK = "I could not find"
SYSTEM_PROMPT = ("You are a POLICY ASSISTANT for Instamart customer support. Answer ONLY from the "
                 "supplied CONTEXT. If the answer is not in the context, reply exactly: "
                 "I could not find that information in the policy documents.")


# ---------------- Task 1: chunking ----------------
def chunk_words(words, size=CHUNK_WORDS, overlap=OVERLAP):
    """Windows of `size` words, each starting `size - overlap` words after the previous one;
    stop after the first window whose end reaches or passes the last word."""
    step, start, chunks = size - overlap, 0, []
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            return chunks
        start += step


def load_chunks(size=CHUNK_WORDS, overlap=OVERLAP):
    chunks = []
    for name in sorted(os.listdir(POLICIES)):
        if name.endswith(".txt"):
            with open(os.path.join(POLICIES, name), encoding="utf-8") as f:
                words = f.read().split()
            chunks += [{"source": name, "text": t} for t in chunk_words(words, size, overlap)]
    return chunks


# ---------------- Task 2: TF-IDF retrieval ----------------
class PolicyIndex:
    """TF-IDF index over the policy chunks (fitted once on the chunk texts)."""

    def __init__(self, size=CHUNK_WORDS, overlap=OVERLAP):
        self.chunks = load_chunks(size, overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        """Top-k chunks by cosine similarity: [{'source', 'score', 'text'}, ...]."""
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        best = scores.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(scores[i]), 4),
                 "text": self.chunks[i]["text"]} for i in best]


_DEFAULT = None


def default_index():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PolicyIndex()
    return _DEFAULT


def retrieve(query, k=3):
    return default_index().retrieve(query, k)


# ---------------- Task 3: augment and answer ----------------
def build_user_message(chunks, question):
    """CONTEXT:\\n[source]\\nchunk ... \\nQUESTION:\\nquestion -- the exact format the brief asks for."""
    context = "\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
    return f"CONTEXT:\n{context}\nQUESTION:\n{question}"


def answer(question, index=None):
    chunks = (index or default_index()).retrieve(question, k=3)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(chunks, question)}]
    reply = mock_llm.chat(messages).choices[0].message.content
    return {"answer": reply, "sources": [c["source"] for c in chunks]}


# ---------------- Task 4: evaluation ----------------
def evaluate(index, answer_fn=answer):
    qa = pd.read_csv(os.path.join("data", "raw", "policy_eval.csv"), keep_default_na=False)
    rows = []
    for q in qa.itertuples(index=False):
        out = answer_fn(q.question, index)
        if q.expected_doc == "NONE":
            rows.append({"question": q.question, "answerable": False, "doc_hit": None,
                         "fact_hit": out["answer"].startswith(FALLBACK), "answer": out["answer"]})
        else:
            rows.append({"question": q.question, "answerable": True,
                         "doc_hit": q.expected_doc in out["sources"],
                         "fact_hit": q.expected_fact in out["answer"], "answer": out["answer"]})
    return pd.DataFrame(rows)


def summarise(ev):
    a, u = ev[ev["answerable"]], ev[~ev["answerable"]]
    return {"doc_in_top3": int(a["doc_hit"].sum()), "fact_in_answer": int(a["fact_hit"].sum()),
            "answerable": len(a), "fallback": int(u["fact_hit"].sum()), "unanswerable": len(u)}


def main():
    print("## Task 1: chunks (40 words, overlap 10)\n")
    index = default_index()
    counts = pd.Series([c["source"] for c in index.chunks]).value_counts()
    print(counts.to_string(), f"\ntotal {len(index.chunks)}")
    assert len(index.chunks) == 15 and counts.to_dict() == {
        "refund_policy.txt": 4, "delivery_policy.txt": 3, "escalation_policy.txt": 3,
        "payment_policy.txt": 3, "privacy_policy.txt": 2}

    print("\n## Task 2: retrieve('refund policy late delivery', k=3)\n")
    top = retrieve("refund policy late delivery")
    for c in top:
        print(f"  {c['score']:.4f}  {c['source']}  {c['text'][:60]}...")
    assert all(c["source"] == "refund_policy.txt" for c in top)

    print("\n## Task 3: example grounded answer\n")
    out = answer("Who must approve a refund above INR 1000?")
    print(f"  answer:  {out['answer']}\n  sources: {out['sources']}")

    print("\n## Task 4: evaluation on data/raw/policy_eval.csv\n")
    results = {}
    for label, idx in [("overlap 10", index), ("overlap 0", PolicyIndex(overlap=0))]:
        ev = evaluate(idx)
        results[label] = (ev, summarise(ev))
    print("| chunking | chunks | expected doc in top 3 | expected fact in answer | fallback on unanswerable |")
    print("|---|---:|---:|---:|---:|")
    for label, idx_chunks in [("overlap 10", 15), ("overlap 0", len(load_chunks(overlap=0)))]:
        s = results[label][1]
        print(f"| {label} | {idx_chunks} | {s['doc_in_top3']}/{s['answerable']} | "
              f"{s['fact_in_answer']}/{s['answerable']} | {s['fallback']}/{s['unanswerable']} |")
    s10 = results["overlap 10"][1]
    assert (s10["doc_in_top3"], s10["fact_in_answer"], s10["fallback"]) == (12, 12, 2)

    ev0 = results["overlap 0"][0]
    broken = ev0[ev0["answerable"] & ~(ev0["fact_hit"].astype(bool) & ev0["doc_hit"].astype(bool))]
    print("\nquestions that break with overlap 0:")
    for r in broken.itertuples(index=False):
        print(f"  - {r.question}\n    answer: {r.answer}")
    assert len(broken) >= 1
    print("\nPART D TASKS 1-4 MATCH REFERENCE")


if __name__ == "__main__":
    main()
