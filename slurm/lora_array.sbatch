#!/bin/bash
#SBATCH --job-name=kras_lora
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/lora_%A_%a.out
# One array task = one (split, fold, seed).  3 splits x 5 folds x SEEDS seeds.
#   sbatch $GPU_FLAGS --array=0-44%6 slurm/lora_array.sbatch        (3 seeds -> 45 tasks, 6 at a time)
# Env vars you can set when submitting:  MODE=lora|head|full   RANK=8   SEEDS=3   SEED_OFFSET=0   MODEL=150M|650M
#   extra seeds:  SEEDS=2 SEED_OFFSET=3 sbatch $GPU_FLAGS --array=0-29%6 slurm/lora_array.sbatch   (adds seeds 3,4 -> 5 in total)
CFG="${SLURM_SUBMIT_DIR:-$PWD}/slurm/config.sh"; [ -f "$CFG" ] || CFG="$HOME/kras_esm/slurm/config.sh"; source "$CFG"
cd $PROJ; source $CONDA_SH; conda activate $ENV_NAME
export HF_HUB_OFFLINE=1
MODE=${MODE:-lora}; RANK=${RANK:-8}; SEEDS=${SEEDS:-3}; SEED_OFFSET=${SEED_OFFSET:-0}
SPLITS=(random heldout_mut heldout_pos); NF=5
i=$SLURM_ARRAY_TASK_ID
SPLIT=${SPLITS[$(( i / (NF*SEEDS) ))]}
FOLD=$(( (i / SEEDS) % NF ))
SEED=$(( SEED_OFFSET + i % SEEDS ))
echo "task $i: split=$SPLIT fold=$FOLD seed=$SEED mode=$MODE model=$MODEL rank=$RANK"
python src/train_lora.py --split $SPLIT --fold $FOLD --seed $SEED --model $MODEL --mode $MODE --rank $RANK
