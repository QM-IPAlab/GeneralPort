#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --account=su008-acw799

source ../env/cliport_env/bin/activate
export CLIPORT_ROOT=~/cliport
export PYTHONPATH=$PYTHONPATH:~/cliport

module purge      
module load CUDA/11.7.0
module load GCCcore/10.2.0 Python/3.8.6

python cliport/eval.py model_task=stack-block-pyramid-seq-seen-colors \
                       eval_task=towers-of-hanoi-seq-seen-colors \
                       agent=cliport \
                       mode=test \
                       n_demos=1 \
                       train_demos=1000 \
                       exp_folder=exps \
                       checkpoint_type=test_best \
                       update_results=True \
                       disp=False \
                       record.save_video=True
