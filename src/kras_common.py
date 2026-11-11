"""Shared constants, split logic and metrics for the KRAS + ESM-2 project."""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", message="An input array is constant")
warnings.filterwarnings("ignore", category=RuntimeWarning)
from scipy.stats import spearmanr, pearsonr

# KRAS residues 2-188, copied from the authors' R code (krasddpcams__read_ddG.R).
# We put the native M back at position 1 so list index = residue number - 1.
WT_2_188 = ("TEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKKKKKSKTKCVIM")
WT_SEQ = "M" + WT_2_188
AAS = "ACDEFGHIKLMNPQRSTVWY"

# file-name code in the authors' MoCHI output -> our column name
TARGET_FILES = {
    "weights_Folding.txt": "ddG_fold",
    "weights_Binding_RAF.txt": "ddG_RAF",
    "weights_Binding_RAL.txt": "ddG_RAL",
    "weights_Binding_PI3.txt": "ddG_PI3",
    "weights_Binding_SOS.txt": "ddG_SOS",
    "weights_Binding_K55.txt": "ddG_K55",
    "weights_Binding_K27.txt": "ddG_K27",
}
TARGETS = list(TARGET_FILES.values())
MAIN_TARGETS = ["ddG_fold", "ddG_RAF"]          # the ones we headline
SPLITS = ["random", "heldout_mut", "heldout_pos"]
N_FOLDS = 5
BLOCK = 10                                       # residues per block in heldout_pos


def load_dataset(path="data/kras_ddg.csv"):
    df = pd.read_csv(path)
    for t in TARGETS:
        if t not in df.columns:
            df[t] = np.nan
    return df.reset_index(drop=True)


def block_id(pos, block=BLOCK):
    return (np.asarray(pos) - 1) // block


def group_ids(df, split):
    """Group label per row: rows with the same label are always held out together."""
    if split == "random":
        return np.arange(len(df))
    if split == "heldout_mut":
        return df["mut"].values
    if split == "heldout_pos":
        return block_id(df["pos"].values)
    raise ValueError(split)


def make_folds(df, split, n_folds=N_FOLDS, fold_seed=0):
    """Fold id (0..n_folds-1) per row. Same for every method so results are comparable."""
    rng = np.random.RandomState(fold_seed)
    if split == "random":
        order = rng.permutation(len(df))
        folds = np.empty(len(df), int)
        folds[order] = np.arange(len(df)) % n_folds
        return folds
    if split == "heldout_mut":
        aas = sorted(df["mut"].unique())
        rng.shuffle(aas)
        amap = {a: i % n_folds for i, a in enumerate(aas)}
        return df["mut"].map(amap).values
    if split == "heldout_pos":
        # contiguous blocks of BLOCK residues, interleaved across folds
        return (block_id(df["pos"].values) % n_folds).astype(int)
    raise ValueError(split)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3 or np.std(b[ok]) == 0:
        return np.nan
    return spearmanr(a[ok], b[ok])[0]


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3 or np.std(b[ok]) == 0:
        return np.nan
    return pearsonr(a[ok], b[ok])[0]


def write_preds(rows, path):
    """rows: list of dicts with method, split, fold, seed, variant, target, y_true, y_pred."""
    pd.DataFrame(rows).to_csv(path, index=False)
