"""Extra analyses that make the results more convincing.  Everything is written to results/extras/ (tables) and
results/extras/figures/ (poster-style PNG + PDF).  Runs on CPU, a few minutes.

    python src/extras.py                               # everything that has its inputs
    python src/extras.py --do seeds,robust             # only some  (seeds robust allo blend ablate)
    python src/extras.py --model 150M --n_boot 1000

  seeds    per-seed spread of LoRA / head-only (needs the extra seeds from submit_extras.sh for 5 seeds, works with 3)
  robust   re-score the existing predictions on the high-confidence measurements only (data/kras_ddg_highconf.csv)
  allo     allostery AUROC when the interface / pocket distance cut-off and the hot-spot cut-off are varied
  blend    does averaging LoRA with zero-shot (and frozen) predictions beat either one?   (no tuning, equal weights)
  ablate   LoRA rank, where LoRA is placed, single- vs multi-task, training-set size, full fine-tuning
"""
import argparse, glob, os, sys, traceback
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from kras_common import *
from evaluate import load_preds, compare, seed_avg
import make_figures as mf

OUT = "results/extras"
mf.OUT = os.path.join(OUT, "figures")
SPLIT_NAME = {"random": "Random", "heldout_mut": "New mutation", "heldout_pos": "New position"}
SPLIT_COL = {"random": mf.GRAY[1], "heldout_mut": mf.BLUE, "heldout_pos": mf.YELLOW}
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------------------------------------ scoring helpers
def seed_scores(P, targets=None):
    """Spearman per (method, split, seed): mean over the 5 folds, then mean over targets (same definition as summary.csv)."""
    if targets is not None:
        P = P[P.target.isin(targets)]
    rows = [(m, s, t, sd, f, spearman(g["y_pred"], g["y_true"]))
            for (m, s, t, sd, f), g in P.groupby(["method", "split", "target", "seed", "fold"])]
    R = pd.DataFrame(rows, columns=["method", "split", "target", "seed", "fold", "rho"])
    R = R.groupby(["method", "split", "target", "seed"], as_index=False)["rho"].mean()
    return R.groupby(["method", "split", "seed"], as_index=False)["rho"].mean()


def summarise(R):
    S = R.groupby(["method", "split"])["rho"].agg(mean="mean", sd="std", n_seeds="count").reset_index()
    return S


def ms(S, method, split):
    r = S[(S.method == method) & (S.split == split)]
    return (float(r["mean"].iloc[0]), float(np.nan_to_num(r["sd"].iloc[0]))) if len(r) else (np.nan, 0.0)


def legend_splits(ax, splits):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SPLIT_COL[s], label=SPLIT_NAME[s]) for s in splits], loc="best")


# ------------------------------------------------------------------------------------------------ 1. seeds
def do_seeds(P, a):
    M = mf.method_names(a.model, a.rank)
    R = seed_scores(P)
    keys = ["zeroshot", "frozen", "head", "lora"]
    Rk = R[R.method.isin([M[k][0] for k in keys])]
    Rk.to_csv(f"{OUT}/seeds_per_seed.csv", index=False)
    S = summarise(Rk); S.to_csv(f"{OUT}/seeds_summary.csv", index=False)
    print(S.round(3).to_string(index=False))
    # how often does a single LoRA seed beat zero-shot?
    zs = Rk[Rk.method == M["zeroshot"][0]].set_index("split")["rho"]
    lo = Rk[Rk.method == M["lora"][0]]
    for s in SPLITS:
        v = lo[lo.split == s]["rho"].values
        print(f"{SPLIT_NAME[s]:13s}: LoRA > zero-shot in {int((v > zs[s]).sum())}/{len(v)} seeds  (seed range {v.min():.3f}-{v.max():.3f}, zero-shot {zs[s]:.3f})")
    mf.style(22)
    fig, ax = plt.subplots(figsize=(16.5, 6.2))
    rng = np.random.RandomState(0)
    for j, s in enumerate(SPLITS):
        for i, k in enumerate(keys):
            vals = Rk[(Rk.method == M[k][0]) & (Rk.split == s)]["rho"].values
            x = j + (i - 1.5) * 0.2
            ax.bar(x, vals.mean(), width=0.17, color=M[k][2], alpha=0.35, lw=0)
            ax.scatter(x + rng.uniform(-0.04, 0.04, len(vals)), vals, s=70, color=M[k][2], edgecolor="white", linewidth=1, zorder=3)
            ax.hlines(vals.mean(), x - 0.085, x + 0.085, color=M[k][2], lw=3, zorder=4)
    ax.set_xticks(range(3)); ax.set_xticklabels([SPLIT_NAME[s] for s in SPLITS])
    ax.set_ylabel("Mean Spearman, 7 targets"); ax.set_ylim(0, Rk["rho"].max() * 1.3)
    n_seeds = int(Rk[Rk.method == M["lora"][0]].seed.nunique())
    ax.set_title(f"Each dot is one training seed ({n_seeds} seeds for the trained models)", loc="left", fontsize=22)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=M[k][2], label=M[k][1]) for k in keys], ncol=4, loc="upper right")
    mf.save(fig, "fig_seeds")


# ------------------------------------------------------------------------------------------------ 2. robustness to measurement quality
def do_robust(P, a):
    if not os.path.exists(a.highconf):
        print(f"robust: {a.highconf} not found. Make it with:\n  python src/make_dataset.py --long 41586_2023_6954_MOESM6_ESM.xlsx --min_conf 1 --out {a.highconf}"); return
    hc = load_dataset(a.highconf)
    valid = hc.melt(id_vars="variant", value_vars=TARGETS, var_name="target").dropna(subset=["value"])[["variant", "target"]]
    full_n = P.drop_duplicates(["variant", "target"]).shape[0]
    Ph = P.merge(valid, on=["variant", "target"])
    keep_n = Ph.drop_duplicates(["variant", "target"]).shape[0]
    print(f"robust: kept {keep_n}/{full_n} variant x target measurements ({100*keep_n/full_n:.0f} %)")
    M = mf.method_names(a.model, a.rank)
    methods = [M[k][0] for k in M]
    Sf = summarise(seed_scores(P[P.method.isin(methods)])); Sh = summarise(seed_scores(Ph[Ph.method.isin(methods)]))
    T = Sf.merge(Sh, on=["method", "split"], suffixes=("_all", "_highconf"))
    T["change"] = T["mean_highconf"] - T["mean_all"]
    T.to_csv(f"{OUT}/robust_summary.csv", index=False); print(T.round(3).to_string(index=False))
    rk = []
    for s in SPLITS:
        t = T[T.split == s]; rk.append((s, spearman(t["mean_all"], t["mean_highconf"])))
    print("method-ranking agreement (Spearman of the method means, all vs high-confidence):", {SPLIT_NAME[s]: round(v, 2) for s, v in rk})
    cmp_rows = []
    for other in ("zeroshot", "head", "frozen"):
        r = compare(Ph, M["lora"][0], M[other][0], n_boot=a.n_boot); r.insert(0, "vs", M[other][1]); cmp_rows.append(r)
    C = pd.concat(cmp_rows); C.to_csv(f"{OUT}/robust_compare_lora.csv", index=False)
    print(C.groupby(["vs", "split"])["verdict"].agg(lambda v: ", ".join(f"{k}:{n}" for k, n in v.value_counts().items())).to_string())
    mf.style(22)
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.4), sharey=True)
    keys = ["onehot", "zeroshot", "frozen", "head", "lora"]
    for ax, s in zip(axes, ["heldout_mut", "heldout_pos"]):
        for i, k in enumerate(keys):
            r = T[(T.method == M[k][0]) & (T.split == s)].iloc[0]
            ax.bar(i - 0.2, r["mean_all"], width=0.38, color=M[k][2], alpha=0.45, lw=0)
            ax.bar(i + 0.2, r["mean_highconf"], width=0.38, color=M[k][2], lw=0)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([M[k][1].replace(" ", "\n", 1) for k in keys], fontsize=17)
        ax.set_title(SPLIT_NAME[s], loc="left")
    axes[0].set_ylabel("Mean Spearman, 7 targets")
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=mf.INK2, alpha=0.45, label="all measurements"), Patch(color=mf.INK2, label="high-confidence only")], loc="upper left")
    mf.save(fig, "fig_robust")


# ------------------------------------------------------------------------------------------------ 3. allostery: sensitivity to the definitions
def annotate(pdb, cut):
    atoms = []
    for line in open(pdb):
        if line.startswith(("ATOM", "HETATM")):
            el = line[76:78].strip() or line[12:16].strip()[0]
            if el == "H":
                continue
            atoms.append((line[:6].strip(), line[21], line[17:20].strip(), int(line[22:26]),
                          np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])))
    nuc = [x for x in atoms if x[0] == "HETATM" and x[2] in {"GTP", "GNP", "GDP", "GCP", "G3D", "GSP"}]
    chain = nuc[0][1]
    partner = np.array([x[4] for x in atoms if x[0] == "ATOM" and x[1] != chain])
    nxyz = np.array([x[4] for x in nuc if x[1] == chain])
    rows = {}
    for x in atoms:
        if x[0] == "ATOM" and x[1] == chain:
            r = rows.setdefault(x[3], dict(pos=x[3], is_interface=False, is_pocket=False))
            r["is_interface"] |= bool(np.min(np.linalg.norm(partner - x[4], axis=1)) < cut)
            r["is_pocket"] |= bool(np.min(np.linalg.norm(nxyz - x[4], axis=1)) < cut)
    return pd.DataFrame(rows.values()).set_index("pos")


def auroc(lab, score):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(lab, score) if 0 < lab.sum() < len(lab) else np.nan


def do_allo(P, a):
    if not os.path.exists(a.pdb):
        print(f"allo: {a.pdb} not found"); return
    M = mf.method_names(a.model, a.rank)
    keys = ["zeroshot", "frozen", "head", "lora"]
    A = seed_avg(P[P.method.isin([M[k][0] for k in keys])])
    A = A[(A.split == "heldout_pos") & (A.target == "ddG_RAF")].copy()
    A["pos"] = A["variant"].str[1:-1].astype(int); A["t"] = A["y_true"].abs(); A["p"] = A["y_pred"].abs()
    PP = {k: A[A.method == M[k][0]].groupby("pos")[["t", "p"]].mean() for k in keys}
    cuts, qs = [4.0, 5.0, 6.0, 8.0], [0.7, 0.8, 0.9]
    rng = np.random.RandomState(0); rows = []
    for cut in cuts:
        ann = annotate(a.pdb, cut)
        for q in qs:
            res = {}
            for k in keys:
                pp = PP[k].join(ann, how="inner"); res[k] = pp[~pp.is_interface & ~pp.is_pocket]
            common = res["lora"].index.intersection(res["zeroshot"].index)
            cand = {k: res[k].loc[common] for k in keys}
            lab = (cand["lora"]["t"] >= cand["lora"]["t"].quantile(q)).values
            if len(common) < 15 or lab.sum() < 2:
                continue
            d = []
            for _ in range(a.n_boot):
                i = rng.randint(0, len(common), len(common))
                d.append(auroc(lab[i], cand["lora"]["p"].values[i]) - auroc(lab[i], cand["zeroshot"]["p"].values[i]))
            row = dict(cutoff_A=cut, hotspot_top_fraction=round(1 - q, 2), n_candidates=len(common), n_hotspots=int(lab.sum()),
                       **{f"auroc_{k}": auroc(lab, cand[k]["p"].values) for k in keys})
            row["lora_minus_zeroshot"] = row["auroc_lora"] - row["auroc_zeroshot"]
            row["ci_low"], row["ci_high"] = np.nanpercentile(d, [2.5, 97.5])
            rows.append(row)
    T = pd.DataFrame(rows); T.to_csv(f"{OUT}/allostery_sweep.csv", index=False); print(T.round(3).to_string(index=False))
    mf.style(22)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0), sharey=True)
    for ax, f in zip(axes, sorted(T.hotspot_top_fraction.unique(), reverse=True)):
        t = T[T.hotspot_top_fraction == f]
        for k in keys:
            ax.plot(t.cutoff_A, t[f"auroc_{k}"], marker="o", ms=11, lw=3.5 if k == "lora" else 2.5, color=M[k][2], label=M[k][1])
        ax.axhline(0.5, color=mf.INK2, lw=1.5, ls=":")
        ax.set_xticks(cuts); ax.set_xlabel("Interface / pocket cut-off (Å)")
        ax.set_title(f"Top {int(round(f*100))} % are hot-spots", loc="left", fontsize=22)
    axes[0].set_ylabel("AUROC, far-from-site positions"); axes[0].set_ylim(0.3, 1.0)
    axes[0].legend(loc="lower left", fontsize=17)
    mf.save(fig, "fig_allo_sweep")


# ------------------------------------------------------------------------------------------------ 4. blend with zero-shot
def do_blend(P, a):
    M = mf.method_names(a.model, a.rank)
    mm = [M["lora"][0], M["zeroshot"][0], M["frozen"][0]]
    A = mf.seed_avg_fold(P[P.method.isin(mm)])
    A["r"] = A.groupby(["method", "split", "target", "fold"])["y_pred"].rank(pct=True)
    W = A.pivot_table(index=["split", "target", "fold", "variant"], columns="method", values="r").dropna().reset_index()
    Y = A[A.method == mm[0]][["split", "target", "variant", "y_true"]].drop_duplicates()
    W = W.merge(Y, on=["split", "target", "variant"])
    L, Z, F = W[mm[0]], W[mm[1]], W[mm[2]]
    variants = {"blend_lora_zeroshot": 0.5 * L + 0.5 * Z, "blend_lora_zeroshot_frozen": (L + Z + F) / 3}
    for w in (0.0, 0.25, 0.75, 1.0):
        variants[f"w{w}"] = w * L + (1 - w) * Z                     # weight curve (for the plot only; nothing is tuned on it)
    rows = []
    for name, v in variants.items():
        rows.append(W[["split", "target", "fold", "variant", "y_true"]].assign(method=name, seed=0, y_pred=v.values))
    for m in mm:                                                      # seed-averaged singles, scored the same way
        rows.append(A[A.method == m][["split", "target", "fold", "variant", "y_true", "method", "y_pred"]].assign(seed=0))
    B = pd.concat(rows, ignore_index=True)
    B[B.method.str.startswith("blend")].to_csv(f"{OUT}/blend_preds.csv", index=False)
    S = summarise(seed_scores(B)); S.to_csv(f"{OUT}/blend_summary.csv", index=False)
    print(S.pivot(index="method", columns="split", values="mean").round(3).to_string())
    C = pd.concat([compare(B, "blend_lora_zeroshot", m, n_boot=a.n_boot).assign(vs=m) for m in (mm[0], mm[1])])
    C.to_csv(f"{OUT}/blend_compare.csv", index=False)
    print(C[["vs", "split", "target", "spearman_diff", "ci_low", "ci_high", "verdict"]].round(3).to_string(index=False))
    mf.style(22)
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.4), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axes[0]
    names = [(mm[1], M["zeroshot"][1], M["zeroshot"][2]), (mm[0], M["lora"][1], M["lora"][2]),
             ("blend_lora_zeroshot", "LoRA + zero-shot", mf.PITT_BLUE), ("blend_lora_zeroshot_frozen", "LoRA + zero-shot\n+ frozen", mf.PITT_GOLD)]
    for j, s in enumerate(SPLITS):
        for i, (m, lab, col) in enumerate(names):
            v, _ = ms(S, m, s); ax.bar(j + (i - 1.5) * 0.2, v, width=0.18, color=col, label=lab if j == 0 else None, lw=0)
    ax.set_xticks(range(3)); ax.set_xticklabels([SPLIT_NAME[s] for s in SPLITS]); ax.set_ylabel("Mean Spearman, 7 targets")
    ax.set_ylim(0, max(p.get_height() for p in ax.patches) * 1.45); ax.legend(loc="upper left", fontsize=16, ncol=2)
    ax = axes[1]
    ws = [0.0, 0.25, 0.5, 0.75, 1.0]; key = {0.0: "w0.0", 0.25: "w0.25", 0.5: "blend_lora_zeroshot", 0.75: "w0.75", 1.0: "w1.0"}
    for s in ["heldout_mut", "heldout_pos"]:
        ax.plot(ws, [ms(S, key[w], s)[0] for w in ws], marker="o", ms=11, lw=3, color=SPLIT_COL[s], label=SPLIT_NAME[s])
    ax.set_xlabel("Weight on LoRA (rest: zero-shot)"); ax.set_ylabel("Mean Spearman"); ax.legend(fontsize=16)
    mf.save(fig, "fig_blend")


# ------------------------------------------------------------------------------------------------ 5. ablations
HID = {"150M": (640, 30), "650M": (1280, 33), "35M": (480, 12), "8M": (320, 6)}


def lora_params(model, r):
    d, L = HID[model]
    return 4 * d * r * L


def do_ablate(P, a):
    files = glob.glob("results/ablations/*/*.csv")
    if not files:
        print("ablate: no files in results/ablations/ yet (run slurm/submit_extras.sh first)"); return
    Pa = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    M = mf.method_names(a.model, a.rank)
    ref, mod = M["lora"][0], a.model
    seeds = sorted(Pa.seed.unique()); sp = ["heldout_mut", "heldout_pos"]
    base = P[P.seed.isin(seeds) & P.split.isin(sp)]
    Pa = Pa.copy(); Pa.loc[Pa.method.str.contains("_only_"), "method"] = f"{ref}_single"
    Pall = pd.concat([base[base.method.isin([ref, M["head"][0]])], Pa, P[P.method == M["zeroshot"][0]]], ignore_index=True)
    S = summarise(seed_scores(Pall)); S.to_csv(f"{OUT}/ablation_summary.csv", index=False)
    S2 = {t: summarise(seed_scores(Pall, targets=[t])) for t in ("ddG_fold", "ddG_RAF")}
    print(f"ablations use seeds {seeds} and the splits {sp}; reference = {ref} on the same seeds")
    print(S.pivot(index="method", columns="split", values="mean").round(3).to_string())
    mf.style(20)
    fig = plt.figure(figsize=(16.5, 10.5)); gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.28)
    ax = {k: fig.add_subplot(gs[i // 3, i % 3]) for i, k in enumerate("abcde")}
    leg = fig.add_subplot(gs[1, 2]); leg.axis("off")
    zs = {s: ms(S, M["zeroshot"][0], s)[0] for s in sp}

    def line_panel(axx, xs, names, xlabels, xlabel, title):
        for s in sp:
            y = [ms(S, n, s)[0] for n in names]; e = [ms(S, n, s)[1] for n in names]
            axx.errorbar(range(len(xs)), y, yerr=e, marker="o", ms=10, lw=3, capsize=4, color=SPLIT_COL[s])
            axx.axhline(zs[s], color=SPLIT_COL[s], lw=1.8, ls="--", alpha=0.8)
        axx.set_xticks(range(len(xs))); axx.set_xticklabels(xlabels); axx.set_xlabel(xlabel); axx.set_title(title, loc="left", fontsize=20)

    def bar_panel(axx, cats, names, title, Sd=S):
        w = 0.8 / len(names)
        for i, (n, lab, col) in enumerate(names):
            for j, s in enumerate(sp):
                v, e = ms(Sd, n, s)
                axx.bar(j + (i - (len(names) - 1) / 2) * w, v, width=w * 0.92, color=col, lw=0, label=lab if j == 0 else None)
        axx.set_xticks(range(len(sp))); axx.set_xticklabels([SPLIT_NAME[s] for s in sp]); axx.set_title(title, loc="left", fontsize=20)

    ranks = [1, 4, 8, 16, 32]
    names = [f"lora_{mod}_r{r}" if r != a.rank else ref for r in ranks]
    line_panel(ax["a"], ranks, names, [f"{r}\n{lora_params(mod, r)/1e6:.2f}M" for r in ranks], "LoRA rank (trainable weights)", "Rank")
    ax["a"].set_ylabel("Mean Spearman, 7 targets")
    fr = [25, 50, 100]
    line_panel(ax["d"], fr, [f"{ref}_frac25", f"{ref}_frac50", ref], [f"{f} %" for f in fr], "Share of training data used", "Training-set size")
    ax["d"].set_ylabel("Mean Spearman, 7 targets")
    bar_panel(ax["b"], sp, [(ref, "query+value, all layers", mf.BLUE), (f"{ref}_all", "all linear layers", mf.AQUA), (f"{ref}_last4", "last 4 layers", mf.VIOLET)], "Where LoRA goes")
    ax["b"].set_ylabel("Mean Spearman, 7 targets")
    # single vs multi-task: fold and RAF separately
    ax["c"].set_title("Single- vs multi-task", loc="left", fontsize=20)
    for i, (n, lab, col) in enumerate([(ref, "multi-task", mf.BLUE), (f"{ref}_single", "single-task", mf.ORANGE)]):
        k = 0
        for t in ("ddG_fold", "ddG_RAF"):
            for s in sp:
                v, e = ms(S2[t], n, s); ax["c"].bar(k + (i - 0.5) * 0.38, v, width=0.34, color=col, lw=0, label=lab if k == 0 else None); k += 1
    ax["c"].set_xticks(range(4)); ax["c"].set_xticklabels(["Fold\nnew mut.", "Fold\nnew pos.", "RAF1\nnew mut.", "RAF1\nnew pos."], fontsize=15)
    bar_panel(ax["e"], sp, [(M["head"][0], "head only", mf.VIOLET), (ref, "LoRA r=8", mf.BLUE), (f"full_{mod}", "full fine-tuning", mf.PITT_BLUE)], "How much of the model to train")
    ax["e"].set_ylabel("Mean Spearman, 7 targets")
    lo = min(v for v in [ms(S, m, s)[0] for m in S.method.unique() for s in sp] if not np.isnan(v))
    for k, axx in ax.items():
        axx.text(-0.02, 1.12, k, transform=axx.transAxes, fontsize=24, fontweight="bold", va="bottom", ha="right")
    for k in "ad":                                  # line panels: zoom in
        ax[k].set_ylim(max(0, lo - 0.08), None)
    for k in "bce":                                 # bar panels: bars start at zero, headroom for the legend
        ax[k].set_ylim(0, max(p.get_height() for p in ax[k].patches) * 1.5)
        ax[k].legend(fontsize=14, loc="upper left", ncol=1)
    ax["a"].tick_params(axis="x", labelsize=14); ax["c"].tick_params(axis="x", labelsize=13)
    from matplotlib.lines import Line2D
    leg.legend(handles=[Line2D([0], [0], color=SPLIT_COL[s], lw=4, marker="o", label=SPLIT_NAME[s]) for s in sp] +
                       [Line2D([0], [0], color=mf.INK2, lw=2, ls="--", label="zero-shot ESM-2 (same colour = split)")], loc="center", fontsize=17)
    mf.save(fig, "fig_ablations")


# ------------------------------------------------------------------------------------------------ main
STEPS = {"seeds": do_seeds, "robust": do_robust, "allo": do_allo, "blend": do_blend, "ablate": do_ablate}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--do", default=",".join(STEPS)); ap.add_argument("--model", default="150M"); ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--preds", default="results/preds"); ap.add_argument("--highconf", default="data/kras_ddg_highconf.csv")
    ap.add_argument("--pdb", default="data/6VJJ.pdb"); ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    P = load_preds(a.preds)
    for step in a.do.split(","):
        print(f"\n=================== {step} ===================")
        try:
            STEPS[step](P, a)
        except Exception:
            traceback.print_exc(); print(f"!! {step} failed (the other steps still run)")


if __name__ == "__main__":
    main()
