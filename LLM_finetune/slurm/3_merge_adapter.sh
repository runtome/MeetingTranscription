#!/bin/bash
#SBATCH -p gpu                    # GPU partition
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH -N 1                      # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 01:00:00               # Max 1 hour
#SBATCH -J merge_adapter
#SBATCH -o merge_adapter_%j.out
#SBATCH -e merge_adapter_%j.err

PROJECT_PATH=/project/lt200239-thaig/MeetingTranscription

module load Mamba
conda activate llamafactory

export PROJECT_PATH=${PROJECT_PATH}
llamafactory-cli export ${PROJECT_PATH}/LLM_finetune/yaml/3_merge_adapter.config.yaml
