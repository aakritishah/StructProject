# Shared by submit_extras.sh and ablation_array.sbatch: which settings belong to which experiment.
# Every experiment = (settings) x (splits) x (5 folds) x (seeds).  Predictions go to results/ablations/<EXP>/.
ABL_SPLITS=(heldout_mut heldout_pos)      # the two splits that test generalisation (random is skipped to save GPU time)
ABL_SEEDS=${ABL_SEEDS:-2}                 # seeds 0..ABL_SEEDS-1 (the main r=8 runs already have seeds 0,1,2 -> used as the reference)
ABL_NF=5
abl_configs () {   # sets CFGS=(...) for experiment $1
  case "$1" in
    rank)   CFGS=("--rank 1 --alpha 2" "--rank 4 --alpha 8" "--rank 16 --alpha 32" "--rank 32 --alpha 64") ;;  # alpha = 2 x rank, like the main r=8 / alpha=16 run
    place)  CFGS=("--modules all" "--modules last4") ;;                      # all attention+MLP layers / last 4 layers only (main run = query+value, all layers)
    single) CFGS=("--only_target ddG_fold" "--only_target ddG_RAF") ;;       # one model per target instead of one multi-task model
    frac)   CFGS=("--train_frac 0.25" "--train_frac 0.5") ;;                 # learning curve (100 % = main run)
    full)   CFGS=("--mode full --lr 2e-5") ;;                                # full fine-tuning of all 150M weights
    *) echo "unknown experiment $1 (rank|place|single|frac|full)" >&2; return 1 ;;
  esac
}
abl_ntasks () { abl_configs "$1" && echo $(( ${#CFGS[@]} * ${#ABL_SPLITS[@]} * ABL_NF * ABL_SEEDS )); }
