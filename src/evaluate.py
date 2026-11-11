"""Collect every predictions file in results/preds and make the summary table + figure.

  python src/evaluate.py                                   # table (results/summary.csv) + figure
  python src/evaluate.py --compare lora_150M_r8 onehot     # paired bootstrap: is LoRA better than one-hot?
  python src/evaluate.py --allostery data/positions.csv    # does it find allosteric positions? (see annotate_structure.py)
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from kras_common import *


def load_preds(d="results/preds"):
    files = sorted(glob.glob(os.path.join(d, "*.csv")))
    if not files:
        sys.exit(f"no prediction files in {d}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def summarise(P):
    rows = []
    for (m, s, t, seed), g in P.groupby(["method", "split", "target", "seed"]):
        per_fold = [spearman(h["y_pred"], h["y_true"]) for _, h in g.groupby("fold")]
        # headline = mean of per-fold Spearman (pooling folds would mix fold-specific offsets)
        rows.append(dict(method=m, split=s, target=t, seed=seed, n=len(g),
                         spearman=float(np.nanmean(per_fold)) if np.any(~np.isnan(per_fold)) else np.nan,
                         spearman_pooled=spearman(g["y_pred"], g["y_true"]), pearson=pearson(g["y_pred"], g["y_true"]),
                         rmse=float(np.sqrt(np.mean((g["y_pred"] - g["y_true"]) ** 2))),
                         fold_sd=float(np.nanstd(per_fold))))
    R = pd.DataFrame(rows)
    S = R.groupby(["method", "split", "target"]).agg(
        n=("n", "first"), spearman=("spearman", "mean"), spearman_sd_seeds=("spearman", "std"),
        spearman_pooled=("spearman_pooled", "mean"), pearson=("pearson", "mean"), rmse=("rmse", "mean"), fold_sd=("fold_sd", "mean"),
        n_seeds=("seed", "nunique")).reset_index()
    return S


def seed_avg(P):
    """Average predictions over seeds so each variant has one prediction per method/split/target."""
    return P.groupby(["method", "split", "target", "variant"], as_index=False).agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))


def compare(P, m1, m2, n_boot=2000, seed=0):
    A = seed_avg(P)
    rng = np.random.RandomState(seed)
    out = []
    for (s, t), g in A.groupby(["split", "target"]):
        a = g[g.method == m1].set_index("variant"); b = g[g.method == m2].set_index("variant")
        common = a.index.intersection(b.index)
        if len(common) < 30:
            continue
        a, b = a.loc[common], b.loc[common]
        d = []
        for _ in range(n_boot):
            i = rng.randint(0, len(common), len(common))
            d.append(spearman(a["y_pred"].values[i], a["y_true"].values[i]) - spearman(b["y_pred"].values[i], b["y_true"].values[i]))
        d = np.array(d)
        diff = spearman(a["y_pred"], a["y_true"]) - spearman(b["y_pred"], b["y_true"])
        out.append(dict(split=s, target=t, n=len(common), spearman_diff=diff, ci_low=np.nanpercentile(d, 2.5), ci_high=np.nanpercentile(d, 97.5),
                        verdict="better" if np.nanpercentile(d, 2.5) > 0 else ("worse" if np.nanpercentile(d, 97.5) < 0 else "not clearly different")))
    return pd.DataFrame(out)


def plot(S, path="results/summary.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = sorted(S["method"].unique(), key=lambda m: (not m.startswith(("lora", "head", "full")), m))
    fig, axes = plt.subplots(len(MAIN_TARGETS), len(SPLITS), figsize=(4.2 * len(SPLITS), 3.4 * len(MAIN_TARGETS)), sharey=True, squeeze=False)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(methods), 2)))
    for r, t in enumerate(MAIN_TARGETS):
        for c, s in enumerate(SPLITS):
            ax = axes[r, c]; sub = S[(S.target == t) & (S.split == s)].set_index("method")
            for i, m in enumerate(methods):
                if m in sub.index:
                    ax.bar(i, sub.loc[m, "spearman"], yerr=np.nan_to_num(sub.loc[m, "spearman_sd_seeds"]), color=colors[i], capsize=2)
            ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=70, ha="right", fontsize=7)
            ax.set_title(f"{t} | {s}", fontsize=9); ax.axhline(0, color="k", lw=0.5)
            if c == 0:
                ax.set_ylabel("Spearman (held-out)")
    fig.tight_layout(); fig.savefig(path, dpi=200); print("wrote", path)


def allostery(P, pos_csv, target="ddG_RAF", thresh_q=0.8):
    """Position-level check using out-of-fold predictions on the heldout_pos split.
    candidate = not in the binding interface and not in the nucleotide pocket (from annotate_structure.py)
    truth     = candidate positions whose mean |true ddG_bind| is in the top (1-thresh_q) of candidates
    score     = mean |predicted ddG_bind| at that position;   metrics: Spearman across candidates and AUROC for truth."""
    from sklearn.metrics import roc_auc_score
    ann = pd.read_csv(pos_csv).set_index("pos")
    A = seed_avg(P); A = A[(A.split == "heldout_pos") & (A.target == target)].copy()
    A["pos"] = A["variant"].str[1:-1].astype(int); A["a_true"] = A["y_true"].abs(); A["a_pred"] = A["y_pred"].abs()
    rows = []
    for m, g in A.groupby("method"):
        pp = g.groupby("pos")[["a_true", "a_pred"]].mean()
        pp = pp.join(ann[["is_interface", "is_pocket"]], how="inner")
        cand = pp[~pp.is_interface & ~pp.is_pocket]
        if len(cand) < 20:
            continue
        lab = cand["a_true"] >= cand["a_true"].quantile(thresh_q)
        rows.append(dict(method=m, n_candidate_positions=len(cand), spearman_pos=spearman(cand["a_pred"], cand["a_true"]),
                         auroc_top_allosteric=roc_auc_score(lab, cand["a_pred"]) if 0 < lab.sum() < len(lab) else np.nan))
    return pd.DataFrame(rows).sort_values("auroc_top_allosteric", ascending=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="results/preds")
    ap.add_argument("--compare", nargs=2, metavar=("METHOD_A", "METHOD_B"))
    ap.add_argument("--allostery", metavar="POSITIONS_CSV")
    a = ap.parse_args()
    P = load_preds(a.preds)
    if a.compare:
        r = compare(P, *a.compare); print(r.to_string(index=False)); r.to_csv("results/compare.csv", index=False)
    elif a.allostery:
        r = allostery(P, a.allostery); print(r.to_string(index=False)); r.to_csv("results/allostery.csv", index=False)
    else:
        S = summarise(P); S.to_csv("results/summary.csv", index=False)
        piv = S[S.target.isin(MAIN_TARGETS)].pivot_table(index=["target", "split"], columns="method", values="spearman").round(3)
        print(piv.to_string()); plot(S)
