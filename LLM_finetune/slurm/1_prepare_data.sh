#!/bin/bash
#SBATCH -p gpu                    # GPU partition
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH -N 1                      # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 02:00:00               # Max 2 hours
#SBATCH -J prepare_data
#SBATCH -o prepare_data_%j.out
#SBATCH -e prepare_data_%j.err

PROJECT_PATH=/project/lt200239-thaig/MeetingTranscription

module load Mamba
conda activate llamafactory

# Step 1: Convert CSV pairs to LLaMA Factory JSON format
python ${PROJECT_PATH}/LLM_finetune/prepare_dataset.py \
    --asr_csv ${PROJECT_PATH}/train/asr_output.csv \
    --gt_csv ${PROJECT_PATH}/train/train.csv \
    --output ${PROJECT_PATH}/LLM_finetune/data/asr_correction_train.json

python ${PROJECT_PATH}/LLM_finetune/prepare_dataset.py \
    --asr_csv ${PROJECT_PATH}/val/asr_output.csv \
    --gt_csv ${PROJECT_PATH}/val/val.csv \
    --output ${PROJECT_PATH}/LLM_finetune/data/asr_correction_val.json

# Step 2: Tokenize dataset
export PROJECT_PATH=${PROJECT_PATH}
llamafactory-cli train ${PROJECT_PATH}/LLM_finetune/yaml/1_data_process.config.yaml
