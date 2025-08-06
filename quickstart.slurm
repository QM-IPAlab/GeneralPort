#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3850
#SBATCH --time=01:00:00
#SBATCH --account=su008-acw799

source ../env/cliport_env/bin/activate
export CLIPORT_ROOT=~/cliport
export PYTHONPATH=$PYTHONPATH:~/cliport

module purge      
module load CUDA/11.7.0
module load GCCcore/10.2.0 Python/3.8.6

srun python cliport/demos.py n=10 \
                        task=stack-block-pyramid-seq-seen-colors \
                        mode=test
