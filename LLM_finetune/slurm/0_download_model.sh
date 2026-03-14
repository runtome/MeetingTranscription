#!/bin/bash
#SBATCH -p gpu                    # GPU partition
#SBATCH --gres=gpu:0              # No GPU needed for download
#SBATCH -N 1                      # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 03:00:00               # Max 3 hours (large model download)
#SBATCH -J download_model
#SBATCH -o download_model_%j.out
#SBATCH -e download_model_%j.err

PROJECT_PATH=/project/lt200239-thaig/MeetingTranscription

module load Mamba
conda activate llamafactory

python ${PROJECT_PATH}/LLM_finetune/download_model.py \
    --model Qwen/Qwen3-8B \
    --save_dir ${PROJECT_PATH}/model/qwen3-8b
