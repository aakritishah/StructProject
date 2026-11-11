"""Build data/kras_ddg.csv (one row per single substitution, one column per target).

Two ways to feed it, depending on what you managed to download:

  A) MoCHI weights files from the authors (weights_Folding.txt, weights_Binding_RAF.txt, ...):
       python src/make_dataset.py --mochi_dir path/to/folder

  B) A wide table (csv/xlsx), e.g. Supplementary Table 5 or the MaveDB download:
       python src/make_dataset.py --inspect path/to/table.xlsx          # look at the columns first
       python src/make_dataset.py --wide path/to/table.xlsx --map "ddG_fold=colA,ddG_RAF=colB"
     (the variant column such as G12D is auto-detected; or pass --col_variant / --col_pos / --col_wt / --col_mut)
"""
import argparse, os, re, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from kras_common import WT_SEQ, TARGET_FILES, TARGETS, AAS


def read_any(path, sheet=None):
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, sheet_name=sheet or 0)
    return pd.read_csv(path, sep=None, engine="python")


def inspect(path):
    if path.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(path)
        print("sheets:", xl.sheet_names)
        for s in xl.sheet_names:
            d = xl.parse(s, nrows=5)
            print(f"\n--- sheet {s!r}: columns = {list(d.columns)}")
            print(d.head(3).to_string())
    else:
        d = read_any(path)
        print("columns:", list(d.columns), "| rows:", len(d))
        print(d.head(5).to_string())


def calibrate_offset(pos_ref, wt):
    """Authors' code uses position = Pos_ref + 1. Check that against the sequence; try other offsets if not."""
    best = None
    for off in (1, 0, 2, -1, 3):
        pos = pos_ref + off
        ok = [(1 <= p <= len(WT_SEQ)) and WT_SEQ[p - 1] == w for p, w in zip(pos, wt)]
        frac = float(np.mean(ok))
        if best is None or frac > best[1]:
            best = (off, frac)
        if frac > 0.98:
            return off, frac
    return best


def from_mochi(d):
    out = None
    for fname, tname in TARGET_FILES.items():
        p = os.path.join(d, fname)
        if not os.path.exists(p):
            print(f"  (missing {fname}; {tname} will be empty)")
            continue
        t = pd.read_csv(p, sep=None, engine="python")
        cols = {c.lower(): c for c in t.columns}
        mean_c = next((cols[c] for c in cols if c.startswith("mean")), None)
        std_c = next((cols[c] for c in cols if c.startswith("std")), None)
        if mean_c is None or "id" not in cols or "pos_ref" not in cols:
            sys.exit(f"{fname}: expected columns id, Pos_ref, mean_kcal/mol; found {list(t.columns)}")
        t = t[t[cols["id"]] != "WT"].copy()
        ids = t[cols["id"]].astype(str)
        t["wt"], t["mut"] = ids.str[0], ids.str[-1]
        t["pos_ref"] = t[cols["pos_ref"]].astype(int)
        t = t[t["mut"].isin(list(AAS)) & t["wt"].isin(list(AAS))]
        off, frac = calibrate_offset(t["pos_ref"].values, t["wt"].values)
        print(f"  {fname}: {len(t)} rows, position offset {off:+d}, wild-type match {frac:.1%}")
        if frac < 0.9:
            sys.exit("Wild-type letters do not match the KRAS sequence; check the file/column meaning.")
        t["pos"] = t["pos_ref"] + off
        t = t[t["wt"] == t["pos"].map(lambda p: WT_SEQ[p - 1])]
        keep = t[["pos", "wt", "mut", mean_c] + ([std_c] if std_c else [])].copy()
        keep.columns = ["pos", "wt", "mut", tname] + ([tname.replace("ddG", "sd")] if std_c else [])
        out = keep if out is None else out.merge(keep, on=["pos", "wt", "mut"], how="outer")
    return out


def from_wide(path, mapping, args):
    t = read_any(path, args.sheet)
    if args.col_variant:
        vc = args.col_variant
    else:
        vc = next((c for c in t.columns if t[c].astype(str).str.fullmatch(r"[A-Z]\d+[A-Z]").mean() > 0.5), None)
    if vc is not None:
        v = t[vc].astype(str).str.extract(r"^([A-Z])(\d+)([A-Z])$")
        t["wt"], t["pos"], t["mut"] = v[0], pd.to_numeric(v[1]), v[2]
    elif args.col_pos and args.col_wt and args.col_mut:
        t["wt"], t["pos"], t["mut"] = t[args.col_wt], t[args.col_pos], t[args.col_mut]
    else:
        sys.exit("Could not find a variant column like G12D. Use --col_variant, or --col_pos/--col_wt/--col_mut.")
    t = t.dropna(subset=["wt", "pos", "mut"])
    t["pos"] = t["pos"].astype(int)
    t = t[t["mut"].isin(list(AAS)) & t["wt"].isin(list(AAS))]
    frac = np.mean([WT_SEQ[p - 1] == w for p, w in zip(t["pos"], t["wt"]) if 1 <= p <= len(WT_SEQ)])
    print(f"  wild-type match with KRAS sequence: {frac:.1%}")
    if frac < 0.9:
        sys.exit("Wild-type letters do not match; positions may be offset in this table.")
    out = t[["pos", "wt", "mut"]].copy()
    for tname, col in mapping.items():
        out[tname] = pd.to_numeric(t[col], errors="coerce")
    return out.groupby(["pos", "wt", "mut"], as_index=False).mean(numeric_only=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mochi_dir"); ap.add_argument("--wide"); ap.add_argument("--inspect")
    ap.add_argument("--map", help='e.g. "ddG_fold=colA,ddG_RAF=colB"'); ap.add_argument("--sheet")
    ap.add_argument("--col_variant"); ap.add_argument("--col_pos"); ap.add_argument("--col_wt"); ap.add_argument("--col_mut")
    ap.add_argument("--out", default="data/kras_ddg.csv")
    a = ap.parse_args()
    if a.inspect:
        return inspect(a.inspect)
    if a.mochi_dir:
        df = from_mochi(a.mochi_dir)
    elif a.wide and a.map:
        mapping = dict(kv.split("=", 1) for kv in a.map.split(","))
        df = from_wide(a.wide, mapping, a)
    else:
        ap.error("give --mochi_dir, or --wide plus --map, or --inspect")
    df["variant"] = df["wt"] + df["pos"].astype(str) + df["mut"]
    df = df[df["wt"] != df["mut"]].sort_values(["pos", "mut"]).reset_index(drop=True)
    for t in TARGETS:
        if t not in df.columns:
            df[t] = np.nan
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}: {len(df)} variants, {df['pos'].nunique()} positions")
    print(df[TARGETS].notna().sum().to_string())


if __name__ == "__main__":
    main()
