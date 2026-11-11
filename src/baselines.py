"""All non-LoRA methods, evaluated on the same folds as the LoRA model.

  pos_mean     average of the training mutations at that position (needs the position to be seen)
  onehot       one-hot wild-type aa + mutant aa -> ridge            (the baseline David asked for)
  onehot_pos   the same plus a one-hot for the position -> ridge   (cannot work on unseen positions)
  props        amino-acid properties + BLOSUM62 -> gradient boosting
  zeroshot     -(ESM-2 log-likelihood ratio), no training
  frozen_ridge frozen ESM-2 embeddings -> ridge
"""
import argparse, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(__file__))
from kras_common import *

KD = dict(zip("ARNDCQEGHILKMFPSTWYV", [1.8,-4.5,-3.5,-3.5,2.5,-3.5,-3.5,-0.4,-3.2,4.5,3.8,-3.9,1.9,2.8,-1.6,-0.8,-0.7,-0.9,-1.3,4.2]))
VOL = dict(zip("ARNDCQEGHILKMFPSTWYV", [88.6,173.4,114.1,111.1,108.5,143.8,138.4,60.1,153.2,166.7,166.7,168.6,162.9,189.9,112.7,89.0,116.1,227.8,193.6,140.0]))
CHG = {a: 0.0 for a in AAS}; CHG.update({"K": 1, "R": 1, "D": -1, "E": -1, "H": 0.1})
ARO = {a: float(a in "FWYH") for a in AAS}
try:
    from Bio.Align import substitution_matrices
    B62 = substitution_matrices.load("BLOSUM62")
except Exception:
    B62 = None


def onehot(letters):
    m = np.zeros((len(letters), 20)); idx = {a: i for i, a in enumerate(AAS)}
    for i, l in enumerate(letters):
        m[i, idx[l]] = 1
    return m


def feats(df, kind):
    if kind == "onehot":
        return np.hstack([onehot(df["wt"]), onehot(df["mut"])])
    if kind == "onehot_pos":
        P = np.zeros((len(df), len(WT_SEQ)))
        P[np.arange(len(df)), df["pos"].values - 1] = 1
        return np.hstack([feats(df, "onehot"), P])
    if kind == "props":
        cols = []
        for tab in (KD, VOL, CHG, ARO):
            w = df["wt"].map(tab).values; m = df["mut"].map(tab).values
            cols += [w, m, m - w]
        b = [B62[w][m] if B62 is not None else 0.0 for w, m in zip(df["wt"], df["mut"])]
        return np.column_stack(cols + [b])
    raise ValueError(kind)


def fit_predict(method, tr, te, y_tr, ctx):
    """tr/te are DataFrames (index = row ids in the full frame)."""
    if method == "pos_mean":
        m = pd.Series(y_tr, index=tr.index).groupby(tr["pos"]).mean()
        return te["pos"].map(m).fillna(np.mean(y_tr)).values
    if method in ("onehot", "onehot_pos"):
        mdl = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(feats(tr, method), y_tr)
        return mdl.predict(feats(te, method))
    if method == "props":
        mdl = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=200, random_state=0)
        mdl.fit(feats(tr, "props"), y_tr)
        return mdl.predict(feats(te, "props"))
    if method == "zeroshot":
        return -ctx["llr"].loc[te["variant"]].values
    if method.startswith("frozen_ridge"):
        X = ctx["emb"]; sc = StandardScaler().fit(X[tr.index])
        mdl = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(sc.transform(X[tr.index]), y_tr)
        return mdl.predict(sc.transform(X[te.index]))
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/kras_ddg.csv")
    ap.add_argument("--model", default=None, help="ESM-2 size whose zero-shot/embedding files to use, e.g. 650M")
    ap.add_argument("--out", default="results/preds/baselines.csv")
    a = ap.parse_args()
    df = load_dataset(a.data)
    methods = ["pos_mean", "onehot", "onehot_pos", "props"]
    ctx = {}
    if a.model:
        zs = f"data/zeroshot_{a.model}.csv"; em = f"data/emb_{a.model}.npz"
        if os.path.exists(zs):
            ctx["llr"] = pd.read_csv(zs).set_index("variant")["llr"]; methods.append("zeroshot")
        else:
            print("no", zs, "(run zero_shot.py first) - skipping zeroshot")
        if os.path.exists(em):
            e = np.load(em, allow_pickle=True)
            assert list(e["variant"]) == list(df["variant"]), "embedding file does not match dataset order"
            ctx["emb"] = np.hstack([e["h_pos"], e["h_mean"]]).astype(np.float32)
            methods.append(f"frozen_ridge")
        else:
            print("no", em, "(run embed.py first) - skipping frozen_ridge")
    rows = []
    for split in SPLITS:
        folds = make_folds(df, split)
        for k in range(N_FOLDS):
            te_mask = folds == k
            for t in TARGETS:
                y = df[t].values; ok = ~np.isnan(y)
                tr = df[~te_mask & ok]; te = df[te_mask & ok]
                if len(tr) < 50 or len(te) < 5:
                    continue
                for m in methods:
                    pred = fit_predict(m, tr, te, tr[t].values, ctx)
                    name = m if m != "frozen_ridge" else f"frozen_ridge_{a.model}"
                    name = name if m != "zeroshot" else f"zeroshot_{a.model}"
                    rows += [dict(method=name, split=split, fold=k, seed=0, variant=v, target=t, y_true=yt, y_pred=yp)
                             for v, yt, yp in zip(te["variant"], te[t], pred)]
        print("done split", split)
    write_preds(rows, a.out)
    print("wrote", a.out, len(rows), "rows")


if __name__ == "__main__":
    main()
