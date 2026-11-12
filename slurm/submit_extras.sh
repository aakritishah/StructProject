#!/bin/bash
# Submits all extra GPU experiments as Slurm arrays (nothing to allocate by hand).  Run from the project folder on the cluster:
#     bash slurm/submit_extras.sh                    # everything, in priority order
#     bash slurm/submit_extras.sh seeds rank         # only some of them   (names: seeds rank place single frac full)
#     CONC=8 bash slurm/submit_extras.sh             # how many array tasks may run at once (default 6)
# Afterwards (when `squeue -M gpu -u $USER` is empty):   sbatch -M teach -p cpu slurm/extras_analysis.sbatch
CFG="${SLURM_SUBMIT_DIR:-$PWD}/slurm/config.sh"; [ -f "$CFG" ] || CFG="$HOME/kras_esm/slurm/config.sh"; source "$CFG"
cd $PROJ; source slurm/ablation_configs.sh; mkdir -p logs results/preds results/ablations
CONC=${CONC:-6}
WHAT=("$@"); [ ${#WHAT[@]} -eq 0 ] && WHAT=(seeds rank place single frac full)
for w in "${WHAT[@]}"; do
  if [ "$w" = "seeds" ]; then
    # two more seeds (3,4) of the main LoRA and head-only runs on all three splits -> 5 seeds in total: 3 splits x 5 folds x 2 seeds = 30 tasks each
    for MODE_ in lora head; do
      J=$(MODE=$MODE_ SEEDS=2 SEED_OFFSET=3 sbatch --parsable $GPU_FLAGS --job-name=kras_seed_$MODE_ --array=0-29%$CONC slurm/lora_array.sbatch)
      echo "seeds ($MODE_): job $J, 30 tasks"
    done
  else
    N=$(abl_ntasks $w) || continue
    J=$(sbatch --parsable $GPU_FLAGS --job-name=kras_$w --export=ALL,EXP=$w --array=0-$((N-1))%$CONC slurm/ablation_array.sbatch)
    echo "$w: job $J, $N tasks"
  fi
done
echo; echo "watch:  squeue -M gpu -u \$USER"
