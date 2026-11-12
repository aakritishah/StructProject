#!/bin/bash
#SBATCH --job-name=kras_figs
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/figs_%j.out
# CPU job: main tables + poster figures.   FIG_MODEL=150M sbatch -M teach -p cpu slurm/analysis.sbatch
CFG="${SLURM_SUBMIT_DIR:-$PWD}/slurm/config.sh"; [ -f "$CFG" ] || CFG="$HOME/kras_esm/slurm/config.sh"; source "$CFG"
cd $PROJ; source $CONDA_SH; conda activate $ENV_NAME; export MPLBACKEND=Agg
M=${FIG_MODEL:-150M}
python src/evaluate.py; python src/evaluate.py --allostery data/positions.csv
python src/make_figures.py --model $M
rm -rf results/figures_$M; cp -r results/figures results/figures_$M
