"""ESM-2 zero-shot score for every substitution (masked-marginal log-likelihood ratio).
score = log p(mutant aa) - log p(wild-type aa) at the masked position.  Higher = more tolerated.
Output: data/zeroshot_<model>.csv  (columns: variant, llr)"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from kras_common import WT_SEQ, load_dataset
from esm_utils import load_model, get_device, encode, MASK, TOK, batches

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="650M"); ap.add_argument("--data", default="data/kras_ddg.csv")
ap.add_argument("--bs", type=int, default=16)
a = ap.parse_args()

dev = get_device()
df = load_dataset(a.data)
model = load_model(a.model, masked_lm=True).to(dev).eval()
positions = sorted(df["pos"].unique())

logp = {}                                   # pos -> log-prob vector over the vocab
with torch.no_grad():
    for idx in batches(len(positions), a.bs):
        ids = []
        for i in idx:
            x = encode(WT_SEQ); x[positions[i]] = MASK; ids.append(x)
        ids = torch.tensor(ids, device=dev)
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits.float()
        lp = torch.log_softmax(out, -1)
        for k, i in enumerate(idx):
            logp[positions[i]] = lp[k, positions[i]].cpu().numpy()

llr = [logp[r.pos][TOK[r.mut]] - logp[r.pos][TOK[r.wt]] for r in df.itertuples()]
out = pd.DataFrame({"variant": df["variant"], "llr": llr})
path = f"data/zeroshot_{a.model}.csv"
out.to_csv(path, index=False)
print("wrote", path, len(out))
