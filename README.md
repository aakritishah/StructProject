# KRAS mutation effects with ESM-2 + a small trainable module

Question: can ESM-2 plus a small parameter-efficient module (LoRA adapters + a small head) predict how single
KRAS mutations change folding stability, binding (6 partners) and allosteric signalling, better than
(a) ESM-2 zero-shot, (b) frozen ESM-2 embeddings, (c) simple baselines (one-hot etc.), on mutations and
positions it has never seen?

## What is in here
```
src/make_dataset.py       build data/kras_ddg.csv from the Weng et al. 2024 data
src/zero_shot.py          ESM-2 zero-shot score for every mutation
src/embed.py              frozen ESM-2 embeddings
src/baselines.py          pos_mean, one-hot, one-hot+position, amino-acid properties, zero-shot, frozen+ridge
src/train_lora.py         the model: ESM-2 + LoRA + head  (also --mode head / full for ablations)
src/evaluate.py           summary table, figure, paired comparison, allostery check
src/annotate_structure.py interface / nucleotide-pocket positions from PDB 6VJJ (for allostery)
slurm/                    cluster job files (edit slurm/config.sh first)
```
Three held-out tests (5 folds each): `random` (sanity), `heldout_mut` (an entire mutant amino acid is hidden, e.g. every
X->W), `heldout_pos` (blocks of 10 neighbouring residues are hidden). Every method uses the same folds.

## Run order (cluster commands: never on the head node, use srun/sbatch)

**1. On your Mac: get the data.** (Where the real data lives: MaveDB `urn:mavedb:00000115`, or Supplementary Tables 4 and 5
of Weng et al. 2024 on the Nature page. The authors' own GitHub only has code.)
```
python src/make_dataset.py --inspect ~/Downloads/<the supplementary table>.xlsx   # shows the columns
python src/make_dataset.py --wide <file> --map "ddG_fold=<col>,ddG_RAF=<col>,..." # -> data/kras_ddg.csv
```
If you get the authors' MoCHI `weights_*.txt` files instead: `python src/make_dataset.py --mochi_dir <folder>`.
Paste me the output of `--inspect` if the column names are confusing and I will give you the exact command.

**2. Copy the folder to the cluster** (from your Mac, in the folder that contains `project/`):
```
rsync -av --exclude results --exclude logs project/ aas701@teach.crc.pitt.edu:kras_esm/
```
(for cluster.csb.pitt.edu use `aas701@cluster.csb.pitt.edu:kras_esm/`)

**3. On the cluster, once:** get a compute node, build the environment, download the models.
```
srun -M gpu -p l40s --gres=gpu:1 --time=01:00:00 --pty bash
cd ~/kras_esm && bash slurm/setup_struct.sh     # adds a few packages to your existing "struct" env, ~3 min
bash slurm/download_models.sh                    # needs internet; see the note inside if it fails
exit
```

**4. Quick test on one fold** (a few minutes) - do this before launching the big array:
```
srun -M gpu -p l40s --gres=gpu:1 --time=00:30:00 --pty bash
cd ~/kras_esm && source ~/miniconda3/etc/profile.d/conda.sh && conda activate struct
python src/train_lora.py --split heldout_pos --fold 0 --epochs 3 --model 35M
```

**5. Launch everything:**
```
cd ~/kras_esm && bash slurm/submit_all.sh          # features + baselines (about 10 min)
# when that has finished:
sbatch -M gpu -p l40s --gres=gpu:1 --array=0-44%6 slurm/lora_array.sbatch                   # the proposal (LoRA)
MODE=head sbatch -M gpu -p l40s --gres=gpu:1 --array=0-44%6 slurm/lora_array.sbatch         # ablation: head only
squeue -M gpu -u $USER
```

**6. Results** (on the cluster, any node):
```
python src/evaluate.py                                              # results/summary.csv + results/summary.png
python src/evaluate.py --compare lora_150M_r8 onehot                # is LoRA really better than one-hot?
python src/annotate_structure.py data/6VJJ.pdb data/positions.csv   # PDB 6VJJ downloaded on your Mac and copied over
python src/evaluate.py --allostery data/positions.csv
```
Copy `results/` back to your Mac with rsync.

## Poster figures (template: template_36x42.pptx, 36 x 42 in)
```bash
python src/make_figures.py            # on a compute node, from ~/kras_esm; ~5 min (the bootstrap is the slow part)
python src/make_figures.py --only main,scatter      # or just some figures
python src/make_figures.py --model 650M              # same figures for the 650M runs
```
Writes `results/figures/fig_*.png|pdf` (300 dpi, drawn at poster-panel size, place at 100 %) and the PyMOL files in
`results/pymol/`. Figures: splits, model, data, training, main, targets, scatter, landscape, profile, forest, allostery.
PyMOL (on your Mac, from the project folder): `@results/pymol/kras_setup.pml`, rotate to a good view,
then `@results/pymol/kras_render.pml` (writes `results/figures/pymol_*.png`). Needs `data/6VJJ.pdb`.

## Notes
* Use `MODEL=150M` in `slurm/config.sh` while testing. For the final numbers switch to `650M` (about 4x slower).
* Do not tune anything on the held-out folds: pick rank / learning rate using the `random` split or the inner validation
  split that `train_lora.py` already uses for early stopping, then report all three splits once.
* Real dataset is small (about 3,200 single substitutions); expect the held-out-position result to be the hard one.

## Extra experiments (robustness + ablations)

All GPU jobs are Slurm arrays; nothing to allocate by hand.

```bash
# on the cluster, in ~/kras_esm
bash slurm/submit_extras.sh                      # submits everything (or name some: seeds rank place single frac full)
squeue -M gpu -u $USER                           # empty = finished
sbatch -M teach -p cpu slurm/extras_analysis.sbatch   # tables + figures -> results/extras/ and results/figures_150M/
```

| name | what | tasks | output |
|---|---|---|---|
| seeds | seeds 3,4 for LoRA and head-only (5 seeds total), all splits | 2 x 30 | results/preds/ |
| rank | LoRA rank 1, 4, 16, 32 (main run = 8) | 80 | results/ablations/rank/ |
| place | LoRA on all linear layers / last 4 layers only | 40 | results/ablations/place/ |
| single | one model per target (folding, RAF1) vs the multi-task model | 40 | results/ablations/single/ |
| frac | 25 % and 50 % of the training data | 40 | results/ablations/frac/ |
| full | full fine-tuning of all weights | 20 | results/ablations/full/ |

Ablations use the two generalisation splits (new mutation, new position) and seeds 0 and 1; the reference is the main
r=8 run restricted to the same seeds.  `src/extras.py` (run by extras_analysis.sbatch) produces fig_seeds, fig_robust
(needs data/kras_ddg_highconf.csv), fig_allo_sweep (needs data/6VJJ.pdb), fig_blend and fig_ablations.
