#!/bin/bash
#SBATCH --job-name=kras_abl
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/abl_%x_%A_%a.out
# One array task = one (setting, split, fold, seed) of ONE ablation experiment. Do not call directly; use:  bash slurm/submit_extras.sh
# needs env var EXP=rank|place|single|frac|full  (submit_extras.sh sets it)
CFG="${SLURM_SUBMIT_DIR:-$PWD}/slurm/config.sh"; [ -f "$CFG" ] || CFG="$HOME/kras_esm/slurm/config.sh"; source "$CFG"
cd $PROJ; source slurm/ablation_configs.sh; source $CONDA_SH; conda activate $ENV_NAME
export HF_HUB_OFFLINE=1
abl_configs "$EXP" || exit 1
i=$SLURM_ARRAY_TASK_ID
PER_CFG=$(( ${#ABL_SPLITS[@]} * ABL_NF * ABL_SEEDS ))
C=$(( i / PER_CFG )); R=$(( i % PER_CFG ))
SPLIT=${ABL_SPLITS[$(( R / (ABL_NF * ABL_SEEDS) ))]}
FOLD=$(( (R / ABL_SEEDS) % ABL_NF ))
SEED=$(( R % ABL_SEEDS ))
echo "task $i: exp=$EXP split=$SPLIT fold=$FOLD seed=$SEED model=$MODEL settings=${CFGS[$C]}"
python src/train_lora.py --split $SPLIT --fold $FOLD --seed $SEED --model $MODEL ${CFGS[$C]} --out_dir results/ablations/$EXP
