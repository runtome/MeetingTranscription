#!/bin/bash
#SBATCH -p gpu				# Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16          		# Specify number of nodes and processors per task
#SBATCH --gpus-per-node=1               # Specify the number of GPUs
#SBATCH --ntasks-per-node=1		# Specify tasks per node
#SBATCH -t 100:00:00			# Specify maximum time limit (hour: minute: second)
#SBATCH -A zz991010			# Specify project name
#SBATCH -J prepare_data 	# Specify job name
#SBATCH -o prepare_data_%j.out		# Specify output file name with job ID
#SBATCH -e prepare_data_%j.err

PROJECT_PATH=/project/zz991000-zdeva/zz991010/MeetingTranscription

module load Miniforge3/25.3.0-3
conda activate /project/zz991000-zdeva/zz991010/llamafactory

export HF_HOME=/project/zz991000-zdeva/zz991010/hf/misc
export HF_DATASETS_CACHE=/project/zz991000-zdeva/zz991010/hf/datasets
export TRANSFORMERS_CACHE=/project/zz991000-zdeva/zz991010/hf/models
export HF_EVALUATE_CACHE=/project/zz991000-zdeva/zz991010/hf/evaluate
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

START=`date`
starttime=$(date +%s)

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
