#!/bin/bash
# Run ONCE on a compute node (not the login node). Adds the few extra packages we need to your existing
# "struct" conda env. It does not reinstall torch. If pip changes your torch version it tells you how to undo it.
#   srun -M gpu -p l40s --gres=gpu:1 --time=00:30:00 --pty bash
#   cd ~/kras_esm && bash slurm/setup_struct.sh
set -e
CFG="${SLURM_SUBMIT_DIR:-$PWD}/slurm/config.sh"; [ -f "$CFG" ] || CFG="$HOME/kras_esm/slurm/config.sh"; source "$CFG"
source $CONDA_SH
conda activate $ENV_NAME
echo "env: $CONDA_DEFAULT_ENV | $(python --version)"
TV1=$(python -c "import torch; print(torch.__version__)")
echo "torch before: $TV1"
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
pip freeze > $HOME/${ENV_NAME}_freeze_before_kras.txt        # backup list of what was installed
pip install transformers peft scikit-learn pandas scipy matplotlib openpyxl biopython
TV2=$(python -c "import torch; print(torch.__version__)")
echo "torch after:  $TV2"
if [ "$TV1" != "$TV2" ]; then
  echo "WARNING: torch changed. To undo:  pip install torch==${TV1%%+*}   (backup list: ~/${ENV_NAME}_freeze_before_kras.txt)"
fi
python - <<'PY'
import torch, transformers, peft, sklearn, pandas
print("transformers", transformers.__version__, "| peft", peft.__version__, "| sklearn", sklearn.__version__)
print("cuda available:", torch.cuda.is_available())
PY
echo "done. Next: bash slurm/download_models.sh"
