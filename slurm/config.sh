# ---- edit this file once, everything else reads it ----
# Where the project lives on the cluster (we rsync the folder there)
export PROJ=$HOME/kras_esm
# your existing conda env (extra packages are added by slurm/setup_struct.sh)
export ENV_NAME=struct
export CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh
# ESM-2 size: use 150M while developing, 650M for the final runs
export MODEL=150M

# ---- Pitt CRC (what worked for Boltz/BoltzGen): GPU jobs on the "gpu" cluster ----
export GPU_FLAGS="-M gpu -p l40s --gres=gpu:1"        # or: -p a100

# ---- if you use cluster.csb.pitt.edu instead, comment the GPU_FLAGS line above and use: ----
# export GPU_FLAGS="-p dept_gpu --gres=gpu:1"         # dept_gpu is not preempted (any_gpu got cancelled before)
# 