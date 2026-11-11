# Plan: finish by Thu Nov 19  (today is Tue Nov 10)

Deliverable format for Nov 19: ______ (report? slides? both?)  <- fill in; I assumed a short written report + figures

## Tue Nov 10 (today): get the pipeline running
- [ ] Download the Weng data (MaveDB urn:mavedb:00000115 or Supp. Tables 4+5) and run `make_dataset.py`  -> data/kras_ddg.csv
- [ ] Check the dataset: how many variants and positions, how many per target (this answers David's data question)
- [ ] rsync the folder to the cluster, build the `kras` env, download ESM-2 150M + 650M
- [ ] Run the one-fold test (README step 4)

## Wed Nov 11
- [ ] **11:59am: submit the 1-slide proposal** (question, methods paragraph, figure, 2 links)
- [ ] `submit_all.sh` -> zero-shot, frozen embeddings, all baselines (150M)
- [ ] Look at the baseline table: sanity check (position-mean and one-hot should do well on `random`)

## Thu Nov 12 - Fri Nov 13: main model
- [ ] LoRA array, 150M, 3 splits x 5 folds x 3 seeds
- [ ] Head-only ablation (same array with MODE=head)
- [ ] First results table (`evaluate.py`). Decide whether to change rank / lr (using only random + inner validation!)

## Sat Nov 14 - Sun Nov 15: final runs + analysis
- [ ] Final model size (650M) for LoRA and head-only, 2-3 seeds
- [ ] Paired comparisons: LoRA vs one-hot, vs zero-shot, vs frozen embeddings
- [ ] Allostery: annotate 6VJJ, run `--allostery`

## Mon Nov 16: figures
- [ ] Main figure: Spearman by method x split (folding and RAF1 binding)
- [ ] One supplementary table with all targets
- [ ] Optional: scatter of predicted vs true for the best model

## Tue Nov 17 - Wed Nov 18: write-up
- [ ] Intro (question), methods (data, model, splits, baselines), results, limitations
- [ ] Limitations to state honestly: small dataset, one protein, position-blocks only 10 residues wide, allostery defined from predictions

## Thu Nov 19: buffer + submit

## If something slips (cut in this order)
1. 650M final runs (report 150M)
2. Binding partners other than RAF1 (report folding + RAF1 only)
3. Allostery analysis (keep as a short discussion)
