"""Poster figures + PyMOL export.  Run from the project folder (on the cluster, after the jobs have finished):

    python src/make_figures.py                 # everything -> results/figures/*.png|pdf, results/pymol/*
    python src/make_figures.py --only main,scatter
    python src/make_figures.py --model 650M    # same figures for the 650M runs

Sizes match the 36 x 42 in template: full-width panel ~16.5 in wide, so figures are drawn at that size with
poster-scale fonts (place them at 100 %, do not shrink).  PNG (300 dpi) and PDF are written for every figure.

Figures
  splits      how the three train/test splits hold data out (positions x amino acids)         -> Methods
  model       model breakdown schematic (ESM-2 + LoRA + multi-task head)                       -> Methods
  data        distribution of the 7 measured targets and how many variants each has           -> Background
  training    train / validation loss per epoch (parsed from logs/)                            -> Results
  main        Spearman per method and split, folding and RAF1 binding (dots = folds)           -> Results
  targets     heat-map: every method x every target on new positions / new mutations          -> Results
  scatter     predicted vs measured on new positions, zero-shot vs LoRA                        -> Results
  landscape   position x amino-acid maps of measured vs predicted effects                      -> Results
  profile     per-residue measured vs predicted effect along the sequence                      -> Results
  forest      paired-bootstrap difference LoRA - baseline with 95 % CI                         -> Results
  allostery   ROC and AUROC (position bootstrap) for far-from-site positions                   -> Results
  pymol       results/pymol/kras_values.csv + kras_figures.pml (run this one in PyMOL on your Mac)
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import ListedColormap
sys.path.insert(0, os.path.dirname(__file__))
from kras_common import *
from evaluate import load_preds, compare

OUT = "results/figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE, AQUA, VIOLET, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100"
GRAY = ["#c9c8c1", "#a5a49d", "#84837c", "#e0dfd9"]          # simple baselines (neutral)
PITT_BLUE, PITT_GOLD = "#003593", "#FFB81C"

TARGET_LABEL = {"ddG_fold": "Folding stability", "ddG_RAF": "RAF1 binding", "ddG_RAL": "RALGDS binding",
                "ddG_PI3": "PI3Kγ binding", "ddG_SOS": "SOS1 binding", "ddG_K55": "DARPin K55 binding",
                "ddG_K27": "DARPin K27 binding"}
SPLIT_LABEL = {"random": "Random", "heldout_mut": "New mutation", "heldout_pos": "New position"}


def method_names(model, rank):
    """key -> (method string in the predictions files, label, colour)"""
    return {
        "onehot": ("onehot", "One-hot", GRAY[0]),
        "onehot_pos": ("onehot_pos", "One-hot + position", GRAY[1]),
        "pos_mean": ("pos_mean", "Position mean", GRAY[2]),
        "props": ("props", "AA properties", GRAY[3]),
        "zeroshot": (f"zeroshot_{model}", "Zero-shot ESM-2", ORANGE),
        "frozen": (f"frozen_ridge_{model}", "Frozen embeddings", AQUA),
        "head": (f"head_{model}", "Head only", VIOLET),
        "lora": (f"lora_{model}_r{rank}", "LoRA (ours)", BLUE),
    }


def style(base=22):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": base, "axes.titlesize": base + 2, "axes.labelsize": base, "xtick.labelsize": base - 3,
        "ytick.labelsize": base - 3, "legend.fontsize": base - 3, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 1.2, "axes.axisbelow": True,
        "pdf.fonttype": 42, "figure.dpi": 100, "legend.frameon": False})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.join(OUT, name + ".png"))


def panel_letter(ax, s):
    ax.text(-0.02, 1.08, s, transform=ax.transAxes, fontsize=26, fontweight="bold", va="bottom", ha="right")


# --------------------------------------------------------------------------------------------- helpers on predictions
def seed_avg_fold(P):
    return P.groupby(["method", "split", "target", "variant"], as_index=False).agg(
        fold=("fold", "first"), y_true=("y_true", "first"), y_pred=("y_pred", "mean"))


def perfold_table(P):
    """Spearman per (method, split, target, fold), averaged over seeds (same definition as results/summary.csv)."""
    rows = []
    for (m, s, t, seed, f), g in P.groupby(["method", "split", "target", "seed", "fold"]):
        rows.append(dict(method=m, split=s, target=t, seed=seed, fold=f, rho=spearman(g["y_pred"], g["y_true"])))
    R = pd.DataFrame(rows)
    return R.groupby(["method", "split", "target", "fold"], as_index=False)["rho"].mean()


def variant_pos(v):
    return int(v[1:-1])


# --------------------------------------------------------------------------------------------- 1. splits
def fig_splits(df, args):
    style(22)
    aas = sorted(AAS)
    pos_all = np.arange(df["pos"].min(), df["pos"].max() + 1)
    fig, axes = plt.subplots(3, 1, figsize=(16.5, 9.5), sharex=True)
    cmap = ListedColormap(["white", "#d9d8d2", ORANGE])
    for ax, split in zip(axes, SPLITS):
        folds = make_folds(df, split)
        M = np.zeros((len(aas), len(pos_all)))
        for r, f in zip(df.itertuples(), folds):
            M[aas.index(r.mut), r.pos - pos_all[0]] = 2 if f == 0 else 1
        ax.imshow(M, cmap=cmap, vmin=0, vmax=2, aspect="auto", interpolation="nearest",
                  extent=[pos_all[0] - .5, pos_all[-1] + .5, len(aas) - .5, -.5])
        ax.set_yticks(range(0, len(aas), 4)); ax.set_yticklabels([aas[i] for i in range(0, len(aas), 4)])
        ax.set_ylabel("Mutant aa")
        ax.set_title(SPLIT_LABEL[split], loc="left", fontsize=24, fontweight="bold")
        ax.grid(False)
    axes[-1].set_xlabel("KRAS residue number")
    fig.legend(handles=[Patch(color="#d9d8d2", label="training data"), Patch(color=ORANGE, label="held out (test fold 0)")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, "fig_splits")


# --------------------------------------------------------------------------------------------- 2. model schematic
def parse_param_line(model, mode="lora"):
    for f in sorted(glob.glob(os.environ.get("KRAS_LOGS", "logs") + "/lora_*.out")):
        txt = open(f, errors="ignore").read()
        if f"model={model}" in txt and f"mode={mode}" in txt:
            m = re.search(r"(\d[\d,]*) trainable backbone params of ([\d,]+) \(([\d.]+)%\), head ([\d,]+)", txt)
            if m:
                return [x.replace(",", "") for x in m.groups()]
    return None


def fig_model(args):
    style(22)
    fig, ax = plt.subplots(figsize=(16.5, 8.4)); ax.set_xlim(0, 100); ax.set_ylim(-9, 46); ax.axis("off")

    def box(x, y, w, h, fc, ec=INK2, text="", fs=20, tc=INK, lw=2, **kw):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw, **kw))
        if text:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=28, lw=2.5, color=INK2))

    # 1 input sequence
    ax.text(1, 40.5, "Input", fontsize=22, fontweight="bold")
    wt = WT_SEQ[8:20]
    ax.text(1, 31, "KRAS sequence", fontsize=18, color=INK2)
    for i, c in enumerate(wt):
        hl = i == 4
        box(1 + i * 1.9, 24, 1.8, 5, "#ffe3d6" if hl else "#f2f1ed", text=("D" if hl else c), fs=17,
            ec=ORANGE if hl else GRAY[0], tc=ORANGE if hl else INK)
    ax.text(1, 20.5, "one substitution per input\n(e.g. G12D)", fontsize=18, color=INK2, va="top")
    arrow(25.5, 26.5, 31, 26.5)

    # 2 ESM-2 with LoRA
    ax.text(32, 40.5, "ESM-2 language model", fontsize=22, fontweight="bold")
    box(32, 10, 30, 28, "#f2f1ed", ec=GRAY[1], lw=2.5)
    for k in range(3):
        y = 31.5 - k * 8.6
        box(34, y, 26, 5.6, "white", ec=GRAY[1], text="", lw=2)
        ax.text(47, y + 2.8, "frozen attention + MLP block" if k < 2 else "... same in every layer", ha="center", va="center", fontsize=16, color=INK2)
        if k < 2:
            for j, lab in enumerate(["Q", "V"]):
                box(52.2 + j * 3.9, y + 5.0, 3.4, 2.6, BLUE, ec=BLUE, text=lab, fs=14, tc="white", lw=1)
    ax.text(47, 7.5, "LoRA adapters (blue, trainable) on\nattention query and value", ha="center", va="top", fontsize=18, color=BLUE)
    arrow(62.5, 24, 66, 24)

    # 3 head
    ax.text(66.5, 40.5, "Prediction head", fontsize=22, fontweight="bold")
    box(66.5, 27.5, 17, 9.5, "#eee9fb", ec=VIOLET, text="embedding at the\nmutated site +\nmean embedding", fs=15)
    box(66.5, 14.5, 17, 8.5, "#eee9fb", ec=VIOLET, text="small MLP\n(trainable)", fs=17)
    arrow(75, 27.3, 75, 23.2)
    arrow(84, 19, 86.6, 19)
    # 4 outputs
    ax.text(87.5, 40.5, "Output", fontsize=22, fontweight="bold")
    outs = ["folding", "RAF1", "RALGDS", "PI3Kγ", "SOS1", "K55", "K27"]
    for i, o in enumerate(outs):
        y = 35.5 - i * 4.3
        box(87.5, y, 12, 3.5, "#fdeee7" if i else "#ffe3d6", ec=ORANGE, text=f"ΔΔG {o}", fs=14, lw=1.5)
    ax.text(93.5, 4.2, "7 stability / binding\nchanges at once", ha="center", va="top", fontsize=18, color=INK2)

    nums = parse_param_line(args.model)
    if nums:
        bb, tot, pct, hd = nums
        ax.text(1, -7.5, f"{args.model} model: {int(bb):,} of {int(tot):,} backbone weights trained ({float(pct):.2f} %)  |  head {int(hd):,} weights",
                fontsize=17, color=INK2)
    save(fig, "fig_model")


# --------------------------------------------------------------------------------------------- 3. data overview
def fig_data(df, args):
    style(22)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16.5, 6.4), gridspec_kw=dict(width_ratios=[2.3, 1]))
    vals = [df[t].dropna().values for t in TARGETS]
    parts = a1.violinplot(vals, showextrema=False, widths=0.85)
    for b in parts["bodies"]:
        b.set_facecolor(BLUE); b.set_alpha(0.55); b.set_edgecolor("none")
    a1.plot(range(1, len(TARGETS) + 1), [np.median(v) for v in vals], "o", color="white", ms=9, mec=INK2)
    a1.axhline(0, color=INK2, lw=1.2, ls="--")
    a1.set_xticks(range(1, len(TARGETS) + 1))
    a1.set_xticklabels(["Folding", "RAF1", "RALGDS", "PI3K\u03b3", "SOS1", "DARPin\nK55", "DARPin\nK27"], fontsize=17)
    a1.set_ylabel("ΔΔG (kcal/mol)")
    a1.set_title("Measured effect of each substitution", loc="left")
    panel_letter(a1, "a")
    n = [len(v) for v in vals]
    a2.barh(range(len(TARGETS)), n, color=BLUE, height=0.62)
    for i, v in enumerate(n):
        a2.text(v + 40, i, f"{v:,}", va="center", fontsize=18)
    a2.set_yticks(range(len(TARGETS))); a2.set_yticklabels([TARGET_LABEL[t] for t in TARGETS], fontsize=17)
    a2.invert_yaxis(); a2.set_xlim(0, max(n) * 1.25); a2.grid(axis="x"); a2.grid(axis="y", visible=False)
    a2.set_xlabel("Variants measured"); a2.set_title("Training data per target", loc="left")
    panel_letter(a2, "b")
    fig.tight_layout()
    save(fig, "fig_data")


# --------------------------------------------------------------------------------------------- 4. training curves
def parse_logs(model, rank):
    runs = []
    for f in sorted(glob.glob(os.environ.get("KRAS_LOGS", "logs") + "/lora_*.out")):
        txt = open(f, errors="ignore").read()
        h = re.search(r"task \d+: split=(\S+) fold=(\d+) seed=(\d+) mode=(\S+) model=(\S+) rank=(\d+)", txt)
        if not h:
            continue
        split, fold, seed, mode, mdl, rk = h.groups()
        if mdl != model or (mode == "lora" and int(rk) != rank):
            continue
        ep = [(int(a), float(b), float(c)) for a, b, c in re.findall(r"epoch (\d+) train ([\d.]+) val ([\d.]+)", txt)]
        if ep:
            runs.append(dict(split=split, mode=mode, ep=np.array(ep)))
    return runs


def fig_training(args):
    runs = parse_logs(args.model, args.rank)
    if not runs:
        print("training: no usable logs/lora_*.out found, skipping"); return
    style(22)
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.2), sharey=True)
    cols = {"random": BLUE, "heldout_mut": ORANGE, "heldout_pos": AQUA}
    for ax, mode, title in zip(axes, ["lora", "head"], ["LoRA fine-tuning", "Head only (backbone frozen)"]):
        for split in SPLITS:
            rr = [r["ep"] for r in runs if r["mode"] == mode and r["split"] == split]
            if len(rr) < 3:
                continue
            L = max(len(x) for x in rr)
            for j, ls, nm in [(1, "-", "train"), (2, "--", "validation")]:
                mat = np.full((len(rr), L), np.nan)
                for i, x in enumerate(rr):
                    mat[i, :len(x)] = x[:, j]
                keep = (~np.isnan(mat)).sum(0) >= 3
                ep = np.arange(1, L + 1)[keep]; mu = np.nanmedian(mat[:, keep], 0)
                lo, hi = np.nanpercentile(mat[:, keep], [25, 75], axis=0)
                ax.plot(ep, mu, ls, color=cols[split], lw=3.2)
                if j == 2:
                    ax.fill_between(ep, lo, hi, color=cols[split], alpha=0.15, lw=0)
        ax.set_title(title, loc="left"); ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].set_ylabel("Loss (standardised MSE)\nmedian and middle 50 % of runs", fontsize=18)
    axes[0].set_ylim(0, 1.6)
    panel_letter(axes[0], "a"); panel_letter(axes[1], "b")
    h = [plt.Line2D([], [], color=cols[s], lw=4, label=SPLIT_LABEL[s]) for s in SPLITS]
    h += [plt.Line2D([], [], color=INK2, lw=3, ls="-", label="train"), plt.Line2D([], [], color=INK2, lw=3, ls="--", label="validation")]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "fig_training")


# --------------------------------------------------------------------------------------------- 5. main bars
def fig_main(P, M, args):
    style(22)
    PF = perfold_table(P)
    order = [k for k in M if M[k][0] in set(P["method"])]
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.4), sharey=True)
    rng = np.random.RandomState(0)
    for ax, t in zip(axes, MAIN_TARGETS):
        w = 0.8 / len(order)
        for si, s in enumerate(SPLITS):
            for mi, k in enumerate(order):
                sub = PF[(PF.method == M[k][0]) & (PF.split == s) & (PF.target == t)]["rho"].dropna()
                if sub.empty:
                    continue
                x = si + (mi - (len(order) - 1) / 2) * w
                ax.bar(x, sub.mean(), width=w * 0.92, color=M[k][2], zorder=2, lw=0)
                ax.plot(np.full(len(sub), x) + rng.uniform(-w * 0.18, w * 0.18, len(sub)), sub, "o", ms=4.5, color=INK, alpha=0.55, zorder=3)
                if k == "lora":
                    ax.text(x, sub.max() + 0.03, f"{sub.mean():.2f}", ha="center", fontsize=17, fontweight="bold", color=BLUE)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_ylim(-0.1, 0.9)
        ax.set_xticks(range(3)); ax.set_xticklabels([SPLIT_LABEL[s] for s in SPLITS])
        ax.set_title(TARGET_LABEL[t], loc="left")
    axes[0].set_ylabel("Spearman correlation on held-out data")
    panel_letter(axes[0], "a"); panel_letter(axes[1], "b")
    fig.legend(handles=[Patch(color=M[k][2], label=M[k][1]) for k in order], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.12), columnspacing=1.5)
    fig.text(0.5, -0.17, "Bars: mean over 5 folds (and seeds); dots: individual folds. Position mean cannot predict new positions.",
             ha="center", fontsize=16, color=INK2)
    fig.tight_layout()
    save(fig, "fig_main")
    PF.to_csv("results/perfold.csv", index=False)


# --------------------------------------------------------------------------------------------- 6. all targets heat-map
def fig_targets(P, M, args):
    style(22)
    PF = perfold_table(P).groupby(["method", "split", "target"], as_index=False)["rho"].mean()
    order = [k for k in M if M[k][0] in set(P["method"])]
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.2), sharey=True)
    for ax, s in zip(axes, ["heldout_pos", "heldout_mut"]):
        A = np.full((len(order), len(TARGETS)), np.nan)
        for i, k in enumerate(order):
            for j, t in enumerate(TARGETS):
                v = PF[(PF.method == M[k][0]) & (PF.split == s) & (PF.target == t)]["rho"]
                if len(v):
                    A[i, j] = v.iloc[0]
        im = ax.imshow(np.ma.masked_invalid(A), cmap="Blues", vmin=0, vmax=0.8, aspect="auto")
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isnan(A[i, j]):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=14, color=INK2)
                else:
                    ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=17,
                            color="white" if A[i, j] > 0.42 else INK, fontweight="bold" if order[i] == "lora" else "normal")
        ax.set_xticks(range(len(TARGETS))); ax.set_xticklabels([t.replace("ddG_", "") for t in TARGETS], fontsize=18)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([M[k][1] for k in order])
        ax.set_title(SPLIT_LABEL[s], loc="left"); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    panel_letter(axes[0], "a"); panel_letter(axes[1], "b")
    fig.tight_layout()
    save(fig, "fig_targets")


# --------------------------------------------------------------------------------------------- 7. scatter
def fig_scatter(P, M, args):
    style(22)
    A = seed_avg_fold(P)
    PF = perfold_table(P).groupby(["method", "split", "target"], as_index=False)["rho"].mean()
    combos = [("zeroshot", "ddG_fold"), ("lora", "ddG_fold"), ("zeroshot", "ddG_RAF"), ("lora", "ddG_RAF")]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 5.0))
    for ax, (k, t), L in zip(axes, combos, "abcd"):
        g = A[(A.method == M[k][0]) & (A.split == "heldout_pos") & (A.target == t)]
        if g.empty:
            ax.axis("off"); continue
        ax.scatter(g.y_true, g.y_pred, s=16, color=M[k][2], alpha=0.35, lw=0, zorder=2)
        lo, hi = np.percentile(g.y_true, [0.5, 99.5])
        if k == "lora":
            ax.plot([lo, hi], [lo, hi], "--", color=INK2, lw=1.6)
        rho = PF[(PF.method == M[k][0]) & (PF.split == "heldout_pos") & (PF.target == t)]["rho"].iloc[0]
        ax.set_title(f"{M[k][1]}\n{TARGET_LABEL[t]}", fontsize=19, loc="left")
        ax.text(0.04, 0.93, f"ρ = {rho:.2f}", transform=ax.transAxes, fontsize=20, fontweight="bold", va="top")
        ax.set_xlabel("Measured (kcal/mol)", fontsize=17)
        ax.set_ylabel("Predicted" + (" (kcal/mol)" if k != "zeroshot" else " score"), fontsize=17)
        ax.grid(axis="x", visible=False)
        panel_letter(ax, L)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.suptitle("Positions never seen during training", x=0.01, ha="left", y=0.99, fontsize=22, fontweight="bold")
    save(fig, "fig_scatter")


# --------------------------------------------------------------------------------------------- allostery helpers
def position_table(P, df, ann, M, target):
    """Per-residue mean |effect| (measured and predicted, new-position split), joined with structure annotation."""
    A = seed_avg_fold(P)
    A = A[(A.split == "heldout_pos") & (A.target == target)].copy()
    A["pos"] = A["variant"].map(variant_pos); A["a_true"] = A["y_true"].abs(); A["a_pred"] = A["y_pred"].abs()
    out = {}
    for k in M:
        g = A[A.method == M[k][0]]
        if g.empty:
            continue
        out[k] = g.groupby("pos")[["a_true", "a_pred"]].mean()
    return out


def allo_inputs(P, df, ann, M, target="ddG_RAF"):
    from sklearn.metrics import roc_auc_score
    pt = position_table(P, df, ann, M, target)
    fold_eff = df.assign(a=df["ddG_fold"].abs()).groupby("pos")["a"].mean()
    res = {}
    for k, pp in pt.items():
        pp = pp.join(ann[["is_interface", "is_pocket"]], how="inner").join(fold_eff.rename("fold_eff"), how="left")
        pp["is_interface"] = pp["is_interface"].astype(bool); pp["is_pocket"] = pp["is_pocket"].astype(bool)
        cand = pp[~pp.is_interface & ~pp.is_pocket].dropna(subset=["a_pred", "a_true"])
        if len(cand) >= 20 and cand["a_pred"].nunique() > 1:
            res[k] = cand
    return res


def auroc_top(cand, q=0.8):
    from sklearn.metrics import roc_auc_score
    lab = cand["a_true"] >= cand["a_true"].quantile(q)
    if lab.sum() == 0 or lab.sum() == len(lab):
        return np.nan
    return roc_auc_score(lab, cand["a_pred"])


def boot_auroc(cand, n_boot=1000, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_boot):
        s = cand.iloc[rng.randint(0, len(cand), len(cand))]
        out.append(auroc_top(s))
    return np.array(out)


def fig_allostery(P, df, ann, M, args):
    from sklearn.metrics import roc_curve
    style(22)
    C = allo_inputs(P, df, ann, M)
    if "lora" not in C:
        print("allostery: no LoRA predictions for held-out positions, skipping"); return
    keys = [k for k in ["lora", "zeroshot", "head", "frozen", "pos_mean"] if k in C]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16.5, 7.0), gridspec_kw=dict(width_ratios=[1, 1.35]))
    for k in keys:
        cand = C[k]; lab = cand["a_true"] >= cand["a_true"].quantile(0.8)
        fpr, tpr, _ = roc_curve(lab, cand["a_pred"])
        a1.plot(fpr, tpr, color=M[k][2], lw=4 if k == "lora" else 2.6, label=M[k][1], zorder=3 if k == "lora" else 2)
    a1.plot([0, 1], [0, 1], "--", color=INK2, lw=1.5)
    a1.set_xlabel("False-positive rate"); a1.set_ylabel("True-positive rate"); a1.set_aspect("equal")
    a1.set_title("Far-from-site hot-spots", loc="left"); a1.legend(loc="lower right", fontsize=17); a1.grid(axis="x", visible=False)
    panel_letter(a1, "a")
    rows = []
    for k in keys:
        cand = C[k]
        b = boot_auroc(cand)
        low = cand[cand["fold_eff"] <= cand["fold_eff"].median()]         # stability-controlled: positions where folding barely changes
        bl = boot_auroc(low, seed=1) if len(low) >= 20 else np.array([np.nan])
        rows.append(dict(method=M[k][1], key=k, n_all=len(cand), auroc_all=auroc_top(cand), lo_all=np.nanpercentile(b, 2.5), hi_all=np.nanpercentile(b, 97.5),
                         n_ctrl=len(low), auroc_ctrl=auroc_top(low) if len(low) >= 20 else np.nan,
                         lo_ctrl=np.nanpercentile(bl, 2.5), hi_ctrl=np.nanpercentile(bl, 97.5)))
    R = pd.DataFrame(rows)
    if "zeroshot" in C:                                                     # paired position bootstrap LoRA - zero-shot
        j = C["lora"][["a_true", "a_pred"]].join(C["zeroshot"][["a_pred"]], rsuffix="_zs", how="inner")
        rng = np.random.RandomState(2); d = []
        for _ in range(1000):
            s = j.iloc[rng.randint(0, len(j), len(j))]
            lab = s["a_true"] >= s["a_true"].quantile(0.8)
            if 0 < lab.sum() < len(lab):
                from sklearn.metrics import roc_auc_score
                d.append(roc_auc_score(lab, s["a_pred"]) - roc_auc_score(lab, s["a_pred_zs"]))
        print(f"LoRA - zero-shot AUROC difference (position bootstrap): {np.mean(d):.3f}  95% CI [{np.percentile(d, 2.5):.3f}, {np.percentile(d, 97.5):.3f}]")
        R["lora_minus_zeroshot_ci"] = f"[{np.percentile(d, 2.5):.3f}, {np.percentile(d, 97.5):.3f}]"
    os.makedirs("results", exist_ok=True); R.to_csv("results/allostery_bootstrap.csv", index=False)
    x = np.arange(len(R)); w = 0.38
    for off, col, lo, hi, lab, alpha in [(-w / 2, "auroc_all", "lo_all", "hi_all", "all far-from-site positions", 1.0),
                                          (w / 2, "auroc_ctrl", "lo_ctrl", "hi_ctrl", "only positions with little folding effect", 0.55)]:
        cols = [M[k][2] for k in R["key"]]
        a2.bar(x + off, R[col], width=w * 0.92, color=cols, alpha=alpha, zorder=2, lw=0)
        a2.errorbar(x + off, R[col], yerr=[R[col] - R[lo], R[hi] - R[col]], fmt="none", ecolor=INK, capsize=5, lw=2, zorder=3)
    a2.axhline(0.5, color=INK2, ls="--", lw=1.5)
    a2.set_xticks(x); a2.set_xticklabels([m.replace(" (ours)", "\n(ours)").replace("Zero-shot ESM-2", "Zero-shot").replace("Frozen embeddings", "Frozen\nembed.").replace("Head only", "Head\nonly").replace("Position mean", "Position\nmean") for m in R["method"]], fontsize=17)
    a2.set_ylim(0.3, 1.2); a2.set_ylabel("AUROC (95 % CI, bootstrap over positions)", fontsize=18)
    a2.grid(axis="x", visible=False)
    a2.legend(handles=[Patch(fc=INK2, alpha=1.0, label="all far-from-site positions"), Patch(fc=INK2, alpha=0.5, label="stability-controlled")],
              loc="upper left", ncol=2, fontsize=16)
    a2.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    a2.set_title("Ranking positions with strong RAF1-binding effects", loc="left", fontsize=20)
    panel_letter(a2, "b")
    fig.tight_layout()
    save(fig, "fig_allostery")


# --------------------------------------------------------------------------------------------- 8. landscape
def fig_landscape(P, M, args):
    style(22)
    aas = sorted(AAS)
    A = seed_avg_fold(P)
    t = "ddG_RAF"
    sets = [("Measured", None, "y_true", INK), ("LoRA prediction", "lora", "y_pred", BLUE), ("Zero-shot prediction", "zeroshot", "y_pred", ORANGE)]
    sets = [s for s in sets if s[1] is None or M[s[1]][0] in set(A.method)]
    pos_all = np.arange(1, len(WT_SEQ) + 1)
    fig, axes = plt.subplots(len(sets), 1, figsize=(16.5, 3.0 * len(sets) + 1.2), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (title, k, col, c) in zip(axes, sets):
        g = A[(A.split == "heldout_pos") & (A.target == t) & (A.method == (M["lora"][0] if k is None else M[k][0]))].copy()
        if g.empty:
            ax.axis("off"); continue
        g["pos"] = g["variant"].map(variant_pos); g["mut"] = g["variant"].str[-1]
        rk = g[col].rank(pct=True)
        Z = np.full((len(aas), len(pos_all)), np.nan)
        for r, rr in zip(g.itertuples(), rk):
            Z[aas.index(r.mut), r.pos - 1] = rr
        ax.imshow(np.ma.masked_invalid(Z), cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                  extent=[0.5, len(pos_all) + .5, len(aas) - .5, -.5])
        ax.set_yticks(range(0, len(aas), 4)); ax.set_yticklabels([aas[i] for i in range(0, len(aas), 4)])
        ax.set_title(title, loc="left", fontsize=21, color=c, fontweight="bold"); ax.grid(False)
    axes[-1].set_xlabel("KRAS residue number")
    fig.text(0.005, 0.5, "Mutant amino acid", rotation=90, va="center", fontsize=20)
    fig.tight_layout(rect=(0.02, 0, 1, 0.94))
    fig.suptitle("RAF1 binding effect (darker = stronger, ranked within each panel; new-position predictions)", x=0.01, ha="left", y=0.99, fontsize=20)
    save(fig, "fig_landscape")


# --------------------------------------------------------------------------------------------- 9. profile
def fig_profile(P, df, ann, M, args):
    style(22)
    C = {}
    pt = position_table(P, df, ann, M, "ddG_RAF")
    if "lora" not in pt:
        print("profile: no LoRA predictions, skipping"); return
    fig, ax = plt.subplots(figsize=(16.5, 5.2))
    z = lambda s: (s - s.mean()) / s.std()
    pos = pt["lora"].index
    ax.fill_between(pos, 0, z(pt["lora"]["a_true"]), color=GRAY[0], alpha=0.9, lw=0, label="Measured", step="mid")
    ax.plot(pos, z(pt["lora"]["a_pred"]), color=BLUE, lw=3, label="LoRA", zorder=3)
    if "zeroshot" in pt:
        ax.plot(pt["zeroshot"].index, z(pt["zeroshot"]["a_pred"]), color=ORANGE, lw=2.4, label="Zero-shot", zorder=2)
    for col, c, lab in [("is_interface", "#9fc4ee", "RAF1 interface"), ("is_pocket", "#f6d58a", "Nucleotide pocket")]:
        idx = ann.index[ann[col].astype(bool)]
        for i, p in enumerate(idx):
            ax.axvspan(p - .5, p + .5, color=c, alpha=0.55, lw=0, label=lab if i == 0 else None, zorder=1)
    ax.set_xlabel("KRAS residue number"); ax.set_ylabel("Mean |effect| (z-score)")
    ax.set_title("RAF1 binding effect per residue (new-position predictions)", loc="left")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=18)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_profile")


# --------------------------------------------------------------------------------------------- 10. forest
def fig_forest(P, M, args):
    style(22)
    path = "results/compare_all.csv"
    if args.recompute or not os.path.exists(path):
        rows = []
        for k in ["zeroshot", "frozen", "head", "onehot_pos"]:
            if M[k][0] not in set(P["method"]):
                continue
            r = compare(P, M["lora"][0], M[k][0], n_boot=args.n_boot); r["baseline"] = k; rows.append(r)
        pd.concat(rows).to_csv(path, index=False)
    R = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 8.0), sharey=True)
    ks = [k for k in ["zeroshot", "frozen", "head", "onehot_pos"] if k in set(R.baseline)]
    off = np.linspace(-0.27, 0.27, len(ks))
    for ax, s in zip(axes, ["heldout_pos", "heldout_mut"]):
        for ti, t in enumerate(TARGETS):
            for oi, k in zip(off, ks):
                r = R[(R.split == s) & (R.target == t) & (R.baseline == k)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.errorbar(r.spearman_diff, ti + oi, xerr=[[r.spearman_diff - r.ci_low], [r.ci_high - r.spearman_diff]],
                            fmt="o", color=M[k][2], ms=10, capsize=4, lw=2.4, zorder=3)
        ax.axvline(0, color=INK, lw=1.5)
        ax.set_yticks(range(len(TARGETS))); ax.set_yticklabels([TARGET_LABEL[t] for t in TARGETS], fontsize=18)
        ax.invert_yaxis(); ax.set_title(SPLIT_LABEL[s], loc="left"); ax.set_xlabel("Δ Spearman (LoRA − baseline)")
        ax.grid(axis="y", visible=False); ax.grid(axis="x")
    panel_letter(axes[0], "a"); panel_letter(axes[1], "b")
    fig.legend(handles=[plt.Line2D([], [], marker="o", color=M[k][2], lw=2.4, ms=10, label=f"vs {M[k][1]}") for k in ks],
               loc="lower center", ncol=len(ks), bbox_to_anchor=(0.5, -0.07), fontsize=18)
    fig.text(0.5, -0.12, "Right of 0 = LoRA better. Intervals: 95 % paired bootstrap over variants (rows).", ha="center", fontsize=16, color=INK2)
    fig.tight_layout()
    save(fig, "fig_forest")


# --------------------------------------------------------------------------------------------- 11. PyMOL export
def export_pymol(P, df, ann, M, args):
    os.makedirs("results/pymol", exist_ok=True)
    cols = {}
    tabs = []
    for t in ["ddG_RAF", "ddG_fold"]:
        pt = position_table(P, df, ann, M, t)
        if "lora" not in pt:
            continue
        d = pd.DataFrame({f"true_{t}": pt["lora"]["a_true"], f"lora_{t}": pt["lora"]["a_pred"]})
        if "zeroshot" in pt:
            d[f"zeroshot_{t}"] = pt["zeroshot"]["a_pred"]
        tabs.append(d)
    if not tabs:
        print("pymol: no predictions, skipping"); return
    V = pd.concat(tabs, axis=1).join(ann[["is_interface", "is_pocket"]], how="outer").reset_index().rename(columns={"index": "pos"})
    V.to_csv("results/pymol/kras_values.csv", index=False)
    C = allo_inputs(P, df, ann, M)
    top_true = list(C["lora"].index[C["lora"]["a_true"] >= C["lora"]["a_true"].quantile(0.8)]) if "lora" in C else []
    top_lora = list(C["lora"].index[C["lora"]["a_pred"] >= C["lora"]["a_pred"].quantile(0.8)]) if "lora" in C else []
    top_zs = list(C["zeroshot"].index[C["zeroshot"]["a_pred"] >= C["zeroshot"]["a_pred"].quantile(0.8)]) if "zeroshot" in C else []
    sel = lambda ps: "+".join(str(int(p)) for p in sorted(ps)) if len(ps) else "9999"
    SETUP = ["# Generated by src/make_figures.py.  Step 1 in PyMOL (started from the PROJECT FOLDER):   @results/pymol/kras_setup.pml",
             "# Then rotate/zoom to the view you like, then Step 2:   @results/pymol/kras_render.pml   (every picture reuses your view)",
             "# (PDB 6VJJ: chain A = KRAS, chain B = RAF1.)",
             "reinitialize", "bg_color white", "load data/6VJJ.pdb, kras", "remove solvent", "hide everything",
             "set cartoon_fancy_helices, 1", "set antialias, 2", "set ray_shadows, 0",
             "select kras_A, kras and chain A", "select partner, kras and not chain A and polymer",
             "show cartoon, kras_A", "show cartoon, partner", "color gray85, partner", "set cartoon_transparency, 0.55, partner",
             "select ligand_, kras and organic", "show sticks, ligand_", "color gray30, ligand_ and elem C",
             f"select interface_res, kras_A and resi {sel(ann.index[ann['is_interface'].astype(bool)])}",
             f"select pocket_res, kras_A and resi {sel(ann.index[ann['is_pocket'].astype(bool)])}",
             f"select allo_true, kras_A and resi {sel(top_true)}", f"select allo_lora, kras_A and resi {sel(top_lora)}",
             f"select allo_zeroshot, kras_A and resi {sel(top_zs)}",
             "select hits, allo_lora and allo_true",
             "orient kras_A", "zoom kras_A, 8", ""]
    L = ["# Generated by src/make_figures.py.  Step 2: renders every picture to results/figures/pymol_*.png from the CURRENT view.",
         "# Quick previews: change 'ray=1' to 'ray=0' (much faster, no shading).",
         "png results/figures/pymol_00_orientation.png, width=2400, height=2000, dpi=300, ray=1", ""]
    for t in ["ddG_RAF", "ddG_fold"]:
        for pref in ["true", "lora", "zeroshot"]:
            colname = f"{pref}_{t}"
            if colname not in V.columns:
                continue
            s_ = V[["pos", colname]].dropna()
            rk = (s_[colname].rank(pct=True) * 100).round(1)
            L += [f"# ---- {colname}: colour = percentile rank of mean |effect| per residue (white = below median, red = strongest)",
                  "alter kras, b=0"]
            L += [f"alter kras_A and resi {int(p)}, b={v}" for p, v in zip(s_["pos"], rk)]
            L += ["hide spheres, kras_A", "color gray85, kras_A", "spectrum b, white_red, kras_A, minimum=50, maximum=100",
                  f"png results/figures/pymol_{colname}.png, width=2400, height=2000, dpi=300, ray=1", ""]
    L += ["# ---- sites: interface (blue), nucleotide pocket (gold), far-from-site hot-spots as red spheres",
          "color gray85, kras_A", "color marine, interface_res", "color yellow, pocket_res"]
    for nm, sl in [("measured", "allo_true"), ("lora", "allo_lora"), ("zeroshot", "allo_zeroshot"), ("lora_hits", "hits")]:
        L += [f"show spheres, {sl} and name CA", f"set sphere_scale, 0.9, {sl} and name CA", f"color red, {sl}",
              f"png results/figures/pymol_sites_{nm}.png, width=2400, height=2000, dpi=300, ray=1", f"hide spheres, {sl}"]
    L += ["# hits = far-from-site hot-spots that LoRA predicted AND that are truly strong"]
    open("results/pymol/kras_setup.pml", "w").write("\n".join(SETUP) + "\n")
    open("results/pymol/kras_render.pml", "w").write("\n".join(L) + "\n")
    L = SETUP + L
    open("results/pymol/kras_figures.pml", "w").write("\n".join(L) + "\n")
    print("wrote results/pymol/kras_values.csv, kras_setup.pml, kras_render.pml, kras_figures.pml (both in one)")


# --------------------------------------------------------------------------------------------- main
ALL = ["splits", "model", "data", "training", "main", "targets", "scatter", "landscape", "profile", "forest", "allostery", "pymol"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="150M"); ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--data", default="data/kras_ddg.csv"); ap.add_argument("--positions", default="data/positions.csv")
    ap.add_argument("--preds", default="results/preds"); ap.add_argument("--only", default=",".join(ALL))
    ap.add_argument("--n_boot", type=int, default=1000); ap.add_argument("--recompute", action="store_true", help="redo the bootstrap in the forest plot")
    args = ap.parse_args()
    want = set(args.only.split(","))
    M = method_names(args.model, args.rank)
    df = load_dataset(args.data)
    ann = pd.read_csv(args.positions).set_index("pos") if os.path.exists(args.positions) else None
    needs_preds = want & {"main", "targets", "scatter", "landscape", "profile", "forest", "allostery", "pymol"}
    P = load_preds(args.preds) if needs_preds else None
    if P is not None:
        have = sorted(set(P["method"]))
        print("methods found:", have)
        if M["lora"][0] not in have:
            print(f"WARNING: no predictions for {M['lora'][0]} (check --model / --rank)")
    todo = {
        "splits": lambda: fig_splits(df, args), "model": lambda: fig_model(args), "data": lambda: fig_data(df, args),
        "training": lambda: fig_training(args), "main": lambda: fig_main(P, M, args), "targets": lambda: fig_targets(P, M, args),
        "scatter": lambda: fig_scatter(P, M, args), "landscape": lambda: fig_landscape(P, M, args),
        "profile": lambda: fig_profile(P, df, ann, M, args), "forest": lambda: fig_forest(P, M, args),
        "allostery": lambda: fig_allostery(P, df, ann, M, args), "pymol": lambda: export_pymol(P, df, ann, M, args)}
    for name in ALL:
        if name in want:
            if name in ("profile", "allostery", "pymol") and ann is None:
                print(f"{name}: {args.positions} missing (run annotate_structure.py first), skipping"); continue
            print(f"--- {name}")
            try:
                todo[name]()
            except Exception as e:                      # one broken figure should not stop the rest
                import traceback; traceback.print_exc(); print(f"!! {name} failed: {e}")


if __name__ == "__main__":
    main()
