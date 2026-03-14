#!/bin/bash
#SBATCH -p gpu                    # GPU partition
#SBATCH --gres=gpu:4              # Request 4 GPUs
#SBATCH -N 1                      # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -t 24:00:00               # Max 24 hours
#SBATCH -J train_sft
#SBATCH -o train_sft_%j.out
#SBATCH -e train_sft_%j.err

PROJECT_PATH=/project/lt200239-thaig/MeetingTranscription

module load Mamba
conda activate llamafactory

export PROJECT_PATH=${PROJECT_PATH}
FORCE_TORCHRUN=1 llamafactory-cli train ${PROJECT_PATH}/LLM_finetune/yaml/2_lora_sft.config.yaml
