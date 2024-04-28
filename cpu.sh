#!/bin/bash
#
#SBATCH --partition=disc_dual_a100_students,normal
#SBATCH --cpus-per-task=64
#SBATCH --mem=80G
#SBATCH --output=outputs/hw8_%j_stdout.txt
#SBATCH --error=outputs/hw8_%j_stderr.txt
#SBATCH --time=06:00:00
#SBATCH --job-name=hw8
#SBATCH --mail-user=brandondmichaud@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504319/cs5043-hw8

. /home/fagg/tf_setup.sh
conda activate tf


python hw8_base.py -vv @exp.txt @oscer.txt @diffusion.txt @noise.txt --cpus_per_task $SLURM_CPUS_PER_TASK
