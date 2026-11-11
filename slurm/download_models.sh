#!/bin/bash
# Downloads ESM-2 weights into the Hugging Face cache (about 0.6 GB for 150M, 2.6 GB for 650M).
# Run on a compute node if the login node blocks big downloads. Compute nodes sometimes have no internet:
# if this fails, download on your Mac (python -c "from transformers import EsmModel; EsmModel.from_pretrained('facebook/esm2_t30_150M_UR50D')")
# and copy ~/.cache/huggingface to the cluster with rsync.
set -e
CFG="${SLURM_SUBMIT_DIR:-$PWD}/slurm/config.sh"; [ -f "$CFG" ] || CFG="$HOME/kras_esm/slurm/config.sh"; source "$CFG"
source $CONDA_SH; conda activate $ENV_NAME
python - <<'PY'
from transformers import EsmModel, EsmForMaskedLM
for name in ["facebook/esm2_t30_150M_UR50D", "facebook/esm2_t33_650M_UR50D"]:
    EsmModel.from_pretrained(name); EsmForMaskedLM.from_pretrained(name); print("ok", name)
PY
