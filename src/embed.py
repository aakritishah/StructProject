"""Frozen ESM-2 embeddings of every mutant sequence (last layer).
Saves data/emb_<model>.npz with h_pos (hidden state at the mutated residue) and h_mean (mean over residues)."""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from kras_common import WT_SEQ, load_dataset
from esm_utils import load_model, get_device, variant_ids, batches

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="650M"); ap.add_argument("--data", default="data/kras_ddg.csv")
ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args()

dev = get_device()
df = load_dataset(a.data)
model = load_model(a.model).to(dev).eval()
ids_all = torch.tensor([variant_ids(WT_SEQ, r.pos, r.mut) for r in df.itertuples()])
pos_all = torch.tensor(df["pos"].values)
hp, hm = [], []
with torch.no_grad():
    for idx in batches(len(df), a.bs):
        ids = ids_all[idx].to(dev)
        hs = model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state.float()
        hp.append(hs[torch.arange(len(idx)), pos_all[idx].to(dev)].cpu())
        hm.append(hs[:, 1:-1].mean(1).cpu())
path = f"data/emb_{a.model}.npz"
np.savez_compressed(path, variant=df["variant"].values,
                    h_pos=torch.cat(hp).numpy().astype(np.float16),
                    h_mean=torch.cat(hm).numpy().astype(np.float16))
print("wrote", path)
