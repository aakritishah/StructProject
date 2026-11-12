#!/bin/bash
#SBATCH --job-name=kras_extras
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/extras_%j.out
# CPU job (no GPU).  Re-scores everything and makes all extra tables + figures.  Safe to run any time; steps whose inputs are missing are skipped.
#     sbatch -M teach -p cpu slurm/extras_analysis.sbatch                     (150M figures)
CFG="${SLURM_SUBMIT_DIR:-$PWD}/slurm/config.sh"; [ -f "$CFG" ] || CFG="$HOME/kras_esm/slurm/config.sh"; source "$CFG"
cd $PROJ; source $CONDA_SH; conda activate $ENV_NAME; export MPLBACKEND=Agg
M=${FIG_MODEL:-150M}
# high-confidence version of the dataset (only needs the supplementary xlsx once)
if [ ! -f data/kras_ddg_highconf.csv ] && [ -f 41586_2023_6954_MOESM6_ESM.xlsx ]; then
  python src/make_dataset.py --long 41586_2023_6954_MOESM6_ESM.xlsx --min_conf 1 --out data/kras_ddg_highconf.csv
fi
# main tables + poster figures again (they now include the extra seeds)
python src/evaluate.py
python src/evaluate.py --allostery data/positions.csv
python src/make_figures.py --model $M
rm -rf results/figures_$M; cp -r results/figures results/figures_$M
# the extra analyses -> results/extras/
python src/extras.py --model $M
echo "done: results/extras/ and results/figures_$M/"
