#!/bin/bash
#SBATCH --job-name=kras_feat
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/feat_%j.out
# zero-shot scores + frozen embeddings + all baselines (one GPU, a few minutes)
CFG="${SLURM_SUBMIT_DIR:-$PWD}/slurm/config.sh"; [ -f "$CFG" ] || CFG="$HOME/kras_esm/slurm/config.sh"; source "$CFG"
cd $PROJ; source $CONDA_SH; conda activate $ENV_NAME
export HF_HUB_OFFLINE=1
python src/zero_shot.py --model ${MODEL}
python src/embed.py --model ${MODEL}
python src/baselines.py --model ${MODEL} --out results/preds/baselines_${MODEL}.csv
