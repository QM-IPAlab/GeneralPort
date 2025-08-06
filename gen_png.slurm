#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --account=su008-acw799

source ../env/cliport_env/bin/activate
export CLIPORT_ROOT=~/cliport
export PYTHONPATH=$PYTHONPATH:~/cliport

module purge      
module load CUDA/11.7.0
module load GCCcore/10.2.0 Python/3.8.6

srun python cliport/gen_png.py