#!/bin/bash
#
#SBATCH --gres=gpu:1
#SBATCH --partition=disc_dual_a100_students,gpu,gpu_a100
#SBATCH --cpus-per-task=64
#SBATCH --mem=80G
#SBATCH --output=outputs/hw7_%j_stdout.txt
#SBATCH --error=outputs/hw7_%j_stderr.txt
#SBATCH --time=06:00:00
#SBATCH --job-name=hw7
#SBATCH --mail-user=brandondmichaud@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504319/cs5043-hw7

. /home/fagg/tf_setup.sh
conda activate dnn_2024_02
module load cuDNN/8.9.2.26-CUDA-12.2.0


python hw7_base.py -vv --gpu @exp.txt @oscer.txt @discriminator.txt @generator.txt @meta.txt --cpus_per_task $SLURM_CPUS_PER_TASK
