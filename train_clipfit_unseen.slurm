#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --account=su008-acw799

source ../env/torch24_env/bin/activate
export CLIPORT_ROOT=~/cliport
export PYTHONPATH=$PYTHONPATH:~/cliport
export TORCH_USE_SDP_KERNEL=2  # 启用所有可用的优化内核（包括 Flash Attention）

module purge      
module load CUDA/12.4.0
module load GCCcore/12.2.0 Python/3.10.8

srun python cliport/train.py train.task=multi-language-conditioned \
                        train.agent=cliport \
                        train.attn_stream_fusion_type=add \
                        train.trans_stream_fusion_type=conv \
                        train.lang_fusion_type=mult \
                        train.n_demos=100 \
                        train.n_steps=60100 \
                        dataset.cache=False \
                        train.exp_folder=exps \
                        dataset.type=multi 
