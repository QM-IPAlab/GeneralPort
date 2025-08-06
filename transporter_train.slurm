#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --account=su008-acw799

source ../env/torch24_env/bin/activate
export CLIPORT_ROOT=~/cliport
export PYTHONPATH=$PYTHONPATH:~/cliport
export TORCH_USE_SDP_KERNEL=2  # 启用所有可用的优化内核（包括 Flash Attention）

module purge      
module load CUDA/12.4.0
module load GCCcore/12.2.0 Python/3.10.8

srun python cliport/train0.py train.task=align-rope \
                        train.agent=transporter \
                        train.attn_stream_fusion_type=add \
                        train.trans_stream_fusion_type=conv \
                        train.lang_fusion_type=mult \
                        train.n_demos=10 \
                        train.n_steps=820 \
                        train.save_steps=[80,160,400,800] \
                        train.exp_folder=exps_transporter \
                        dataset.cache=False 

srun python cliport/eval0.py model_task=align-rope \
                       eval_task=align-rope \
                       agent=transporter \
                       mode=val \
                       n_demos=25 \
                       train_demos=10 \
                       checkpoint_type=val_missing \
                       exp_folder=exps_transporter

srun python cliport/eval0.py model_task=align-rope \
                       eval_task=align-rope \
                       agent=transporter \
                       mode=test \
                       n_demos=50 \
                       train_demos=10 \
                       checkpoint_type=test_best \
                       exp_folder=exps_transporter  
