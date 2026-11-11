#!/bin/bash
# Run ONCE on a compute node (not the login/head node):
#   srun -M gpu -p l40s --gres=gpu:1 --time=01:00:00 --pty bash      (CRC)
#   cd ~/kras_esm && bash slurm/setup_env.sh
set -e
CFG="${SLURM_SUBMIT_DIR:-$PWD}/slurm/config.sh"; [ -f "$CFG" ] || CFG="$HOME/kras_esm/slurm/config.sh"; source "$CFG"
source $CONDA_SH
conda create -y -n $ENV_NAME python=3.11
conda activate $ENV_NAME
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft scikit-learn pandas scipy matplotlib openpyxl biopython
python - <<'PY'
import torch, transformers, peft
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("transformers", transformers.__version__, "| peft", peft.__version__)
PY
echo "env ready. Next: bash slurm/download_models.sh"
