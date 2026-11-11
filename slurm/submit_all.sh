#!/bin/bash
# Submits the pipeline.  Run from the project folder on the cluster:   bash slurm/submit_all.sh
# (Slurm cannot chain jobs across different clusters, so the LoRA array is submitted separately, after step 1 finishes.)
CFG="${SLURM_SUBMIT_DIR:-$PWD}/slurm/config.sh"; [ -f "$CFG" ] || CFG="$HOME/kras_esm/slurm/config.sh"; source "$CFG"
cd $PROJ; mkdir -p logs results/preds
J1=$(sbatch --parsable $GPU_FLAGS slurm/features.sbatch)
echo "1) features + baselines submitted: $J1"
echo "   when it has finished (squeue -M gpu -u $USER shows nothing), run step 2:"
echo "2) sbatch $GPU_FLAGS --array=0-44%6 slurm/lora_array.sbatch"
echo "   head-only ablation:  MODE=head sbatch $GPU_FLAGS --array=0-44%6 slurm/lora_array.sbatch"
