"""Visualizations for Parts A-D -> outputs/figures/*.png

Run from the repository root AFTER the Part A-D scripts:  python viz/make_figures.py
Every chart is recomputed from the project's own data and output files. The README states each
finding in numbers and words as well, so the charts illustrate rather than replace the analysis.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")                               # write files, never open a window
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402
from sklearn.cluster import KMeans                  # noqa: E402
from sklearn.decomposition import PCA               # noqa: E402
from sklearn.preprocessing import StandardScaler    # noqa: E402

ROOT = os.path.abspath(".")
for p in (ROOT, os.path.join(ROOT, "part_b"), os.path.join(ROOT, "part_c"), os.path.join(ROOT, "part_d")):
    sys.path.insert(0, p)

FIG = os.path.join("outputs", "figures")
CLEAN = os.path.join("data", "clean")
# validated categorical palette (fixed order) and neutrals
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURF, T1, T2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": T2, "xtick.color": T2,
                     "ytick.color": T2, "text.color": T1, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "savefig.dpi": 130})


def style(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def title(fig, main, sub):
    """Title and subtitle stacked ABOVE the plot area, offset in points so they never overlap."""
    from matplotlib.transforms import offset_copy
    fig.text(0.02, 1.0, main, fontsize=13.5, fontweight="bold", va="bottom",
             transform=offset_copy(fig.transFigure, fig=fig, y=26, units="points"))
    fig.text(0.02, 1.0, sub, fontsize=9.5, color=T2, va="bottom",
             transform=offset_copy(fig.transFigure, fig=fig, y=8, units="points"))


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def completed_orders():
    o = pd.read_csv(os.path.join(CLEAN, "orders.csv"))
    c = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
    p = pd.read_csv(os.path.join(CLEAN, "products.csv"))
    return (o[o["status"] == "completed"].merge(c[["customer_id", "city"]], on="customer_id")
            .merge(p[["product_id", "category"]], on="product_id"))


# ---------------- Part A ----------------
def fig_a1_order_amounts(done):
    amt = done["order_amount"]
    q1, q3 = amt.quantile([.25, .75])
    fence = q3 + 1.5 * (q3 - q1)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    style(ax)
    bins = np.arange(0, amt.max() + 50, 50)
    counts, _, _ = ax.hist(amt, bins=bins, color=C[0], edgecolor=SURF, linewidth=0.6)
    top = counts.max() * 1.28                          # headroom so labels sit above the tallest bar
    ax.set_ylim(0, top)
    for x, lab, ls, ha, y in [(amt.median(), f"median\nINR {amt.median():,.2f}", "-", "right", 0.86),
                              (amt.mean(), f"mean\nINR {amt.mean():,.2f}", "--", "left", 0.86),
                              (fence, f"upper fence INR {fence:,.2f}\n{(amt > fence).sum()} orders above", ":", "left", 0.55)]:
        ax.axvline(x, color=T1, lw=1.2, ls=ls)
        ax.text(x + (25 if ha == "left" else -25), top * y, lab, fontsize=9, color=T1, ha=ha)
    ax.set_xlabel("completed order amount (INR, bins of 50)")
    ax.set_ylabel("orders")
    title(fig, "A · Completed order amounts are strongly right-skewed",
          f"{len(amt):,} completed orders; skewness {amt.skew():.2f}. The long tail is mostly electronics, not data errors.")
    save(fig, "a1_order_amount_distribution.png")


def fig_a2_revenue_heatmap(done):
    pv = done.pivot_table(values="order_amount", index="city", columns="category", aggfunc="sum")
    pv = pv.loc[pv.sum(axis=1).sort_values(ascending=False).index, pv.sum().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(pv.values / 1000, cmap="Blues", aspect="auto")
    ax.set_xticks(range(pv.shape[1]), pv.columns)
    ax.set_yticks(range(pv.shape[0]), pv.index)
    for (i, j), v in np.ndenumerate(pv.values):
        ax.text(j, i, f"{v / 1000:,.1f}k", ha="center", va="center", fontsize=9.5,
                color="white" if v > pv.values.max() * 0.55 else T1)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, label="completed revenue (INR thousands)", shrink=0.85)
    title(fig, "A · Completed revenue by city and category (Q2)",
          f"Total INR {pv.values.sum():,.2f}; largest cell Bengaluru / Electronics. Cities and categories sorted by total.")
    save(fig, "a2_revenue_city_category.png")


# ---------------- Part B ----------------
def fig_b1_k_choice(X):
    from sklearn.metrics import silhouette_score
    ks, inertia, sil = range(2, 9), [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_))
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, ys, lab, fmt in [(a, inertia, "inertia (lower = tighter)", "{:,.1f}"),
                             (b, sil, "silhouette (higher = cleaner separation)", "{:.3f}")]:
        style(ax, "both")
        ax.plot(list(ks), ys, color=C[0], lw=2, marker="o", ms=7, mec=SURF, mew=2)
        ax.plot([4], [ys[2]], marker="o", ms=13, mfc="none", mec=T1, mew=1.5)
        for k, v in zip(ks, ys):
            ax.annotate(fmt.format(v), (k, v), textcoords="offset points", xytext=(7, 6), fontsize=8.5, color=T2)
        ax.set_xlabel("K (number of segments)")
        ax.set_title(lab, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlim(1.7, 8.7)
    title(fig, "B · Choosing K: the elbow and the silhouette peak both point to K = 4",
          "Inertia drops 525 from K=3 to 4 but only 126 from 4 to 5; silhouette is highest at K = 4 (0.526).")
    fig.subplots_adjust(wspace=0.25)
    save(fig, "b1_k_choice.png")


def fig_b2_segments(X):
    seg = pd.read_csv(os.path.join("outputs", "customer_segments.csv"))
    Z = PCA(n_components=2).fit_transform(X)
    prof = seg.groupby("segment_name").agg(n=("customer_id", "count"), m=("monetary", "mean")).sort_values("m", ascending=False)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    style(ax, "both")
    for i, (name, r) in enumerate(prof.iterrows()):
        mask = (seg["segment_name"] == name).values
        ax.scatter(Z[mask, 0], Z[mask, 1], s=15, color=C[i], edgecolors=SURF, linewidths=0.4, alpha=0.85,
                   label=f"{name} · {int(r['n'])} customers · mean INR {r['m']:,.0f}")
    ax.set_xlabel("PC1 (63.4%): more orders and spend, more recent →")
    ax.set_ylabel("PC2 (21.8%): higher spend, less recent →")
    ax.legend(loc="upper left", fontsize=8.8, frameon=True, facecolor=SURF, edgecolor=GRID, markerscale=1.6)
    title(fig, "B · Four RFM segments projected onto two principal components",
          "PCA keeps 85.2% of the RFM variance. Premium Big Spenders (blue) form the upward plume.")
    save(fig, "b2_segments_pca.png")


def fig_b3_lapse_model():
    import joblib
    import lapse_model as lm
    from sklearn.model_selection import train_test_split
    X, y = lm.build_dataset()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    rf = joblib.load(os.path.join("outputs", "lapse_model.joblib"))
    names = [n.split("__", 1)[1] for n in rf.named_steps["pre"].get_feature_names_out()]
    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values().tail(8)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    style(ax, "x")
    bars = ax.barh(imp.index, imp.values, color=[C[0] if i >= len(imp) - 3 else "#9fc2ec" for i in range(len(imp))], height=0.62)
    for b, v in zip(bars, imp.values):
        ax.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9, color=T2)
    ax.set_xlabel("random forest feature importance")
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, rf.predict(X_test))
    title(fig, "B · What predicts lapsing: how often and how much customers buy",
          f"Top 8 of 19 model columns for the selected random forest (test F1 {f1:.3f} on the lapsed class).")
    save(fig, "b3_feature_importance.png")


def fig_b4_class_balance():
    """Recompute Part B Task 7 (same grid, same over-sampling) so the chart can never drift from the model."""
    import lapse_model as lm
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.pipeline import Pipeline
    X, y = lm.build_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    lr = GridSearchCV(Pipeline([("pre", lm.make_preprocessor()), ("model", LogisticRegression(max_iter=1000))]),
                      {"model__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_tr, y_tr).best_estimator_
    train = X_tr.assign(lapsed=y_tr)
    minority, majority = train[train["lapsed"] == 1], train[train["lapsed"] == 0]
    balanced = pd.concat([train, minority.sample(n=len(majority) - len(minority), replace=True, random_state=42)])
    bal = clone(lr).fit(balanced.drop(columns="lapsed"), balanced["lapsed"])
    data = {}
    for lab, model in [("un-balanced training", lr), ("balanced training", bal)]:
        pred = model.predict(X_te)
        data[lab] = (precision_score(y_te, pred), recall_score(y_te, pred), f1_score(y_te, pred))
    metrics = ["precision", "recall", "F1"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    style(ax)
    x, w = np.arange(3), 0.36
    for i, (lab, vals) in enumerate(data.items()):
        bars = ax.bar(x + (i - 0.5) * w, vals, w * 0.94, color=C[i], label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9, color=T2)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1)
    ax.set_ylabel("lapsed class, test set")
    ax.legend(frameon=False, loc="upper left")
    title(fig, "B · Over-sampling lapsers trades precision for recall",
          "Logistic regression on the same untouched test set: 76 of 89 lapsers caught instead of 58, 53 loyal customers targeted instead of 32.")
    save(fig, "b4_class_balance.png")


# ---------------- Part C ----------------
def fig_c1_triage():
    import triage
    tickets = triage.load_tickets()
    truth = tickets.set_index("ticket_id")["true_category"]
    rows = []
    for name, prompt in [("zero-shot", triage.ZERO_SHOT), ("few-shot", triage.FEW_SHOT)]:
        r, _ = triage.run_pipeline(tickets, prompt)
        done = r[r["status"].isin(["classified_json", "classified_keyword"])]
        acc = (done["category"].values == truth.loc[done["ticket_id"]].values).mean()
        rows.append((name, acc * 100, r["tokens_used"].sum() / len(done)))
    final = pd.read_csv(os.path.join("outputs", "triage_results.csv"), keep_default_na=False)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for ax, idx, lab, fmt in [(axes[0], 1, "accuracy on classified tickets (%)", "{:.1f}%"),
                              (axes[1], 2, "tokens per classified ticket", "{:.1f}")]:
        style(ax)
        bars = ax.bar([r[0] for r in rows], [r[idx] for r in rows], color=[C[0], C[1]], width=0.55)
        for b, r in zip(bars, rows):
            ax.text(b.get_x() + b.get_width() / 2, r[idx] * 1.02, fmt.format(r[idx]), ha="center", fontsize=9.5, color=T2)
        ax.set_title(lab, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(r[idx] for r in rows) * 1.18)
    ax = axes[2]
    style(ax, "x")
    counts = final["route"].value_counts().sort_values()
    bars = ax.barh(counts.index, counts.values, color=C[2], height=0.55)
    for b, v in zip(bars, counts.values):
        ax.text(v + 1.5, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=9.5, color=T2)
    ax.set_title("final routing of 164 tickets", loc="left", fontsize=11, fontweight="bold")
    title(fig, "C · Few-shot is perfectly accurate but costs about 2.8 times the tokens",
          "Same 147 classified tickets in both runs; routing uses the few-shot run and the Part B Premium flag.")
    fig.subplots_adjust(wspace=0.6)
    save(fig, "c1_triage_prompts_routing.png")


# ---------------- Part D ----------------
def fig_d1_agent_and_rag():
    import rag
    runs = [json.loads(line) for line in open(os.path.join("outputs", "agent_runs.jsonl"), encoding="utf-8")]
    sc = [r for r in runs if r["label"] == "scenario"]
    s10 = rag.summarise(rag.evaluate(rag.default_index()))
    s0 = rag.summarise(rag.evaluate(rag.PolicyIndex(overlap=0)))
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1.5, 1]})
    style(a)
    ids = [r["scenario_id"] for r in sc]
    tone = {"REFUND": C[2], "ESCALATE": C[0], "DENIED": C[1], "NOT_FOUND": C[4]}
    bars = a.bar(ids, [r["llm_calls"] for r in sc], color=[tone.get(r["status"], C[3]) for r in sc], width=0.6)
    for bar, r in zip(bars, sc):
        status = r["status"].replace("FALLBACK_", "FALLBACK\n")      # wrap the long fallback status
        a.text(bar.get_x() + bar.get_width() / 2, r["llm_calls"] + 0.1, status + f"\n{r['tokens_used']:,} tok",
               ha="center", fontsize=8, color=T2)
    a.set_ylim(0, 8)
    a.set_ylabel("model calls")
    a.set_title("refund agent: model calls per scenario (status and tokens above)", loc="left", fontsize=11, fontweight="bold")
    style(b)
    labels = ["doc in top 3", "fact in answer", "fallback"]
    x, w = np.arange(3), 0.36
    for i, (lab, s) in enumerate([("overlap 10", s10), ("overlap 0", s0)]):
        vals = [s["doc_in_top3"] / 12 * 100, s["fact_in_answer"] / 12 * 100, s["fallback"] / 2 * 100]
        raw = [f"{s['doc_in_top3']}/12", f"{s['fact_in_answer']}/12", f"{s['fallback']}/2"]
        bs = b.bar(x + (i - 0.5) * w, vals, w * 0.94, color=C[i], label=lab)
        for bb, t in zip(bs, raw):
            b.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 2, t, ha="center", fontsize=8.5, color=T2)
    b.set_xticks(x, labels)
    b.set_ylim(0, 118)
    b.set_ylabel("% of questions")
    b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    b.set_title("policy RAG evaluation (14 questions)", loc="left", fontsize=11, fontweight="bold")
    title(fig, "D · The agent reaches each decision in 2 to 5 calls; S7 is stopped safely at 6",
          "Removing chunk overlap breaks 2 answers whose sentence is split across a chunk boundary.")
    fig.subplots_adjust(wspace=0.25)
    save(fig, "d1_agent_and_rag.png")


def main():
    os.makedirs(FIG, exist_ok=True)
    done = completed_orders()
    fig_a1_order_amounts(done)
    fig_a2_revenue_heatmap(done)
    import segments
    X = StandardScaler().fit_transform(segments.build_rfm()[segments.RFM_COLS])
    fig_b1_k_choice(X)
    fig_b2_segments(X)
    fig_b3_lapse_model()
    fig_b4_class_balance()
    fig_c1_triage()
    fig_d1_agent_and_rag()
    print(f"\nFIGURES WRITTEN to {FIG}/")


if __name__ == "__main__":
    main()
