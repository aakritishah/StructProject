"""TEST ONLY: writes fake MoCHI-style weights files so the pipeline can be tried without the real data."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from kras_common import WT_SEQ, TARGET_FILES, AAS

out = sys.argv[1] if len(sys.argv) > 1 else "data/fake_mochi"
os.makedirs(out, exist_ok=True)
rng = np.random.RandomState(1)
buried = rng.rand(len(WT_SEQ)) < 0.35
iface = np.zeros(len(WT_SEQ), bool); iface[30:42] = True; iface[55:60] = True
hyd = dict(zip("ACDEFGHIKLMNPQRSTVWY", [1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))
for fname, tname in TARGET_FILES.items():
    rows = []
    for p in range(2, len(WT_SEQ) + 1):
        for a in AAS:
            if a == WT_SEQ[p - 1] or rng.rand() < 0.2:
                continue
            base = (1.5 if buried[p - 1] else 0.2) * (1 + abs(hyd[a] - hyd[WT_SEQ[p - 1]]) / 6)
            if tname != "ddG_fold" and iface[p - 1]:
                base += 1.5
            rows.append({"id": f"{WT_SEQ[p-1]}{p}{a}", "Pos_ref": p - 1,
                         "mean_kcal/mol": base + rng.randn() * 0.3, "std_kcal/mol": 0.1})
    pd.DataFrame([{"id": "WT", "Pos_ref": 0, "mean_kcal/mol": 0, "std_kcal/mol": 0}] + rows).to_csv(
        os.path.join(out, fname), sep="\t", index=False)
print("fake data written to", out)
