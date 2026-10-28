"""Part B, Tasks 1-3 -- RFM table, K-means segments, profiles, PCA.

Run from the repository root:  python part_b/segments.py
Stage 1 (CHOSEN_K = None): builds RFM, scores K = 2..8, and profiles K = 3, 4, 5 so you can choose.
Stage 2 (CHOSEN_K set):    profiles the chosen K, names the segments, runs PCA and writes
                           outputs/customer_segments.csv.
Only completed orders count. Snapshot moment: 2025-10-01 00:00:00. Amounts are in INR.
"""
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

CLEAN, OUT = os.path.join("data", "clean"), "outputs"
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
RFM_COLS = ["recency", "frequency", "monetary"]

CHOSEN_K = 4             # decision: elbow at 4, silhouette peaks at 4 (0.526), and 4 is the
                         # smallest K that separates big spenders from frequent grocery shoppers

# Business name and one recommended action per segment. Segments are identified by their
# PROFILE, not by cluster number -- K-means cluster ids are arbitrary labels with no order.
SEGMENT_ACTIONS = {
    "Premium Big Spenders": "VIP treatment: early access to electronics launches and priority (premium desk) support",
    "Loyal Regulars": "Reward the habit: loyalty points or a subscription on staples, plus basket-building offers",
    "Occasional Shoppers": "Build frequency: free delivery on the next three orders and timely restock reminders",
    "Lapsed Customers": "Win back: a time-limited comeback coupon and a short 'what went wrong' survey",
}


def build_rfm():
    orders = pd.read_csv(os.path.join(CLEAN, "orders.csv"), parse_dates=["order_ts"])
    done = orders[orders["status"] == "completed"]
    rfm = done.groupby("customer_id").agg(last_order=("order_ts", "max"),
                                          frequency=("order_id", "nunique"),
                                          monetary=("order_amount", "sum"))
    rfm["recency"] = (SNAPSHOT - rfm["last_order"]).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)
    return rfm[RFM_COLS].reset_index()


def check_rfm(rfm):
    s = rfm[RFM_COLS].agg(["median", "min", "max", "mean"])
    print("## Task 1: RFM table\n")
    print(f"customers: {len(rfm):,}")
    print(f"recency   median {s.loc['median', 'recency']:.0f}, min {s.loc['min', 'recency']:.0f}, "
          f"max {s.loc['max', 'recency']:.0f}")
    print(f"frequency mean {s.loc['mean', 'frequency']:.2f}, max {s.loc['max', 'frequency']:.0f}")
    print(f"monetary  mean {s.loc['mean', 'monetary']:,.2f}, max {s.loc['max', 'monetary']:,.2f}")
    assert len(rfm) == 1175
    assert (s.loc["median", "recency"], s.loc["min", "recency"], s.loc["max", "recency"]) == (19, 0, 180)
    assert (round(s.loc["mean", "frequency"], 2), s.loc["max", "frequency"]) == (9.46, 33)
    assert (round(s.loc["mean", "monetary"], 2), s.loc["max", "monetary"]) == (2552.06, 14766.55)
    print("RFM matches the reference values")


def score_k(X):
    """Fit K-means for K = 2..8; return inertia and silhouette for each."""
    ref = {2: (1820.8, .465), 3: (1268.2, .445), 4: (743.1, .526), 5: (617.4, .481),
           6: (503.2, .482), 7: (408.5, .416), 8: (366.5, .412)}
    print("\n## Task 2: K-means, K = 2..8\n")
    print("| K | inertia | silhouette | reference | within tolerance |\n|---|---:|---:|---|---|")
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        sil = silhouette_score(X, km.labels_)
        ok = abs(km.inertia_ - ref[k][0]) <= 0.05 * ref[k][0] and abs(sil - ref[k][1]) <= 0.05
        print(f"| {k} | {km.inertia_:,.1f} | {sil:.3f} | {ref[k][0]:,.1f} / {ref[k][1]:.3f} | "
              f"{'yes' if ok else 'NO'} |")


def profile(rfm, labels):
    """Size and mean R, F, M per cluster, largest monetary first."""
    p = (rfm.assign(segment_id=labels).groupby("segment_id")
         .agg(size=("customer_id", "count"), recency=("recency", "mean"),
              frequency=("frequency", "mean"), monetary=("monetary", "mean"))
         .round(1).sort_values("monetary", ascending=False))
    return p


def main():
    rfm = build_rfm()
    check_rfm(rfm)
    X = StandardScaler().fit_transform(rfm[RFM_COLS])
    score_k(X)

    if CHOSEN_K is None:
        print("\n## Candidate profiles (mean recency in days, mean frequency, mean monetary in INR)")
        for k in (3, 4, 5):
            labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
            print(f"\n### K = {k}\n\n{profile(rfm, labels).to_string()}")
        print("\nSTOP: choose K (3, 4 or 5), set CHOSEN_K at the top of this file, and run again.")
        return

    # ---------------- Task 3: profile, name, project ----------------
    labels = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=42).fit_predict(X)
    prof = profile(rfm, labels)
    names = name_segments(prof)
    prof.insert(0, "segment_name", prof.index.map(names))
    premium_id = prof["monetary"].idxmax()            # highest mean monetary, whatever K is
    print(f"\n## Task 3: K = {CHOSEN_K} segment profile (cluster ids are labels only, not an order)\n")
    print("| segment_id | segment_name | size | mean recency (days) | mean frequency | mean monetary (INR) | premium | action |")
    print("|---:|---|---:|---:|---:|---:|---|---|")
    for sid, r in prof.iterrows():
        print(f"| {sid} | {r['segment_name']} | {int(r['size'])} | {r['recency']:.1f} | {r['frequency']:.1f} | "
              f"{r['monetary']:,.1f} | {'yes' if sid == premium_id else 'no'} | {SEGMENT_ACTIONS[r['segment_name']]} |")
    sizes = sorted(prof["size"].astype(int), reverse=True)
    assert sizes == sorted([436, 138, 419, 182], reverse=True), sizes
    assert names[premium_id] == "Premium Big Spenders"

    pca = PCA(n_components=2).fit(X)
    ratio = pca.explained_variance_ratio_
    print(f"\nPCA explained_variance_ratio_: {ratio[0]:.3f} + {ratio[1]:.3f} = {ratio.sum():.3f}")
    assert abs(ratio[0] - 0.634) <= 0.02 and abs(ratio[1] - 0.218) <= 0.02 and abs(ratio.sum() - 0.852) <= 0.02

    out = rfm.assign(segment_id=labels)
    out["segment_name"] = out["segment_id"].map(names)
    out["is_premium"] = out["segment_id"] == premium_id
    os.makedirs(OUT, exist_ok=True)
    out[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]] \
        .to_csv(os.path.join(OUT, "customer_segments.csv"), index=False)
    print(f"\nwrote outputs/customer_segments.csv ({len(out):,} customers, "
          f"{int(out['is_premium'].sum())} premium)")
    print("SEGMENTS MATCH REFERENCE")


def name_segments(prof):
    """Map cluster id -> business name using each segment's profile (works for K = 4)."""
    names = {}
    rest = prof.copy()
    for name, pick in [("Premium Big Spenders", lambda p: p["monetary"].idxmax()),
                       ("Lapsed Customers", lambda p: p["recency"].idxmax()),
                       ("Loyal Regulars", lambda p: p["frequency"].idxmax()),
                       ("Occasional Shoppers", lambda p: p.index[0])]:
        sid = pick(rest)
        names[sid] = name
        rest = rest.drop(sid)
    return names


if __name__ == "__main__":
    main()
