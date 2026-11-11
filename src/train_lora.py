"""ESM-2 + small trainable module -> KRAS ddG (folding + 6 binding partners, multi-task).

  --mode lora   frozen ESM-2 + LoRA adapters on attention (query, value) + small head   <- the proposal
  --mode head   frozen ESM-2 + the same head only (no adapters)                          <- ablation
  --mode full   everything trainable (only sensible for 8M/35M/150M)                     <- ablation

One run = one (split, fold, seed). Writes results/preds/<method>_<split>_f<fold>_s<seed>.csv
"""
import argparse, copy, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from kras_common import *
from esm_utils import load_model, get_device, variant_ids

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, choices=SPLITS)
ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--model", default="150M")
ap.add_argument("--mode", default="lora", choices=["lora", "head", "full"])
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--alpha", type=int, default=16)
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--patience", type=int, default=6)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-4, help="learning rate for backbone params that are trainable")
ap.add_argument("--head_lr", type=float, default=1e-3)
ap.add_argument("--data", default="data/kras_ddg.csv")
ap.add_argument("--out_dir", default="results/preds")
a = ap.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = get_device()
use_bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()

# ---------------------------------------------------------------- data
df = load_dataset(a.data)
folds = make_folds(df, a.split)
test_idx = np.where(folds == a.fold)[0]
trainval_idx = np.where(folds != a.fold)[0]
# inner validation: hold out ~15% of *groups* (same grouping as the outer split) for early stopping
groups = group_ids(df.iloc[trainval_idx], a.split)
ug = np.unique(groups)
rng = np.random.RandomState(a.seed)
val_g = set(rng.choice(ug, size=max(1, int(0.15 * len(ug))), replace=False))
is_val = np.array([g in val_g for g in groups])
val_idx, train_idx = trainval_idx[is_val], trainval_idx[~is_val]

Y = df[TARGETS].values.astype(np.float32)
mu = np.nanmean(Y[train_idx], 0); sd = np.nanstd(Y[train_idx], 0) + 1e-6     # per-target standardisation
Yz = (Y - mu) / sd
M = (~np.isnan(Yz)).astype(np.float32)
Yz = np.nan_to_num(Yz)
IDS = torch.tensor([variant_ids(WT_SEQ, r.pos, r.mut) for r in df.itertuples()])
POS = torch.tensor(df["pos"].values)
Yz_t, M_t = torch.tensor(Yz), torch.tensor(M)

# ---------------------------------------------------------------- model
backbone = load_model(a.model)
if a.mode == "lora":
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, bias="none",
                     target_modules=["query", "value"])
    backbone = get_peft_model(backbone, cfg)
elif a.mode == "head":
    for p in backbone.parameters():
        p.requires_grad = False
D = backbone.config.hidden_size if hasattr(backbone, "config") else backbone.base_model.config.hidden_size


class Head(nn.Module):
    def __init__(s, d, n_out, hid=256, p=0.1):
        super().__init__()
        s.net = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, hid), nn.GELU(), nn.Dropout(p), nn.Linear(hid, n_out))

    def forward(s, h_pos, h_mean):
        return s.net(torch.cat([h_pos, h_mean], -1))


head = Head(D, len(TARGETS))
backbone.to(dev); head.to(dev)
bb_params = [p for p in backbone.parameters() if p.requires_grad]
n_bb = sum(p.numel() for p in bb_params); n_tot = sum(p.numel() for p in backbone.parameters())
print(f"{a.mode}: {n_bb:,} trainable backbone params of {n_tot:,} ({100*n_bb/n_tot:.2f}%), head {sum(p.numel() for p in head.parameters()):,}", flush=True)
groups_opt = [{"params": head.parameters(), "lr": a.head_lr}]
if bb_params:
    groups_opt.append({"params": bb_params, "lr": a.lr})
opt = torch.optim.AdamW(groups_opt, weight_decay=0.01)


def forward(idx, train):
    ids = IDS[idx].to(dev)
    ctx = torch.enable_grad() if (train and bb_params) else torch.no_grad()
    with ctx, torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
        hs = backbone(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
    hs = hs.float()
    h_pos = hs[torch.arange(len(idx)), POS[idx].to(dev)]
    return head(h_pos, hs[:, 1:-1].mean(1))


def run_eval(idx):
    backbone.eval(); head.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(idx), 64):
            outs.append(forward(idx[i:i + 64], False).cpu())
    return torch.cat(outs)


def masked_mse(pred, idx):
    y, m = Yz_t[idx].to(dev), M_t[idx].to(dev)
    return ((pred - y) ** 2 * m).sum() / m.sum().clamp(min=1)


def trainable_state():
    return {k: v.detach().cpu().clone() for k, v in list(backbone.named_parameters()) + [("head." + n, p) for n, p in head.named_parameters()] if v.requires_grad}


def load_state(state):
    with torch.no_grad():
        for k, v in list(backbone.named_parameters()) + [("head." + n, p) for n, p in head.named_parameters()]:
            if k in state:
                v.copy_(state[k].to(v.device))


# ---------------------------------------------------------------- train
best, best_state, bad = 1e9, None, 0
t0 = time.time()
for ep in range(a.epochs):
    backbone.train(bool(bb_params)); head.train()
    perm = np.random.permutation(train_idx)
    tl = 0.0
    for i in range(0, len(perm), a.bs):
        idx = perm[i:i + a.bs]
        loss = masked_mse(forward(idx, True), idx)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups_opt for p in g["params"]], 1.0)
        opt.step(); tl += loss.item() * len(idx)
    with torch.no_grad():
        vl = masked_mse(run_eval(val_idx).to(dev), val_idx).item()
    print(f"epoch {ep+1:02d} train {tl/len(perm):.3f} val {vl:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    if vl < best - 1e-4:
        best, best_state, bad = vl, trainable_state(), 0
    else:
        bad += 1
        if bad >= a.patience:
            print("early stop"); break
load_state(best_state)

# ---------------------------------------------------------------- predict held-out fold
pred = run_eval(test_idx).numpy() * sd + mu
method = f"{a.mode}_{a.model}" + (f"_r{a.rank}" if a.mode == "lora" else "")
rows = []
for j, t in enumerate(TARGETS):
    for k, i in enumerate(test_idx):
        if M[i, j] > 0:
            rows.append(dict(method=method, split=a.split, fold=a.fold, seed=a.seed, variant=df["variant"].iloc[i],
                             target=t, y_true=Y[i, j], y_pred=pred[k, j]))
os.makedirs(a.out_dir, exist_ok=True)
path = os.path.join(a.out_dir, f"{method}_{a.split}_f{a.fold}_s{a.seed}.csv")
write_preds(rows, path)
sp = {t: spearman(*zip(*[(r["y_true"], r["y_pred"]) for r in rows if r["target"] == t])) for t in MAIN_TARGETS}
print("held-out Spearman:", {k: round(v, 3) for k, v in sp.items()}, "->", path)
