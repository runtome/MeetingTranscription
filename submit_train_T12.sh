#!/bin/bash
#SBATCH -p gpu				# Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16          		# Specify number of nodes and processors per task
#SBATCH --gpus-per-node=1               # Specify the number of GPUs
#SBATCH --ntasks-per-node=1		# Specify tasks per node
#SBATCH -t 100:00:00			# Specify maximum time limit (hour: minute: second)
#SBATCH -A zz991010			# Specify project name
#SBATCH -J TrainT12		# Specify job name
#SBATCH -o output_train_T12.txt		# Specify output file name with job ID

module load Miniforge3/25.3.0-3
conda activate /project/zz991000-zdeva/zz991010/env

export HF_HOME=/project/zz991000-zdeva/zz991010/hf/misc
export HF_DATASETS_CACHE=/project/zz991000-zdeva/zz991010/hf/datasets
export TRANSFORMERS_CACHE=/project/zz991000-zdeva/zz991010/hf/models
export HF_EVALUATE_CACHE=/project/zz991000-zdeva/zz991010/hf/evaluate
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

START=`date`
starttime=$(date +%s)

srun   python finetune_vibe_voice.py \
      --model_name microsoft/VibeVoice-ASR-HF \
      --batch_size 2 \
      --gradient_accumulation_steps 8 \
      --learning_rate 1e-4 \
      --warmup_ratio 0.1 \
      --epochs 3 \
      --lora_r 16 \
      --lora_alpha 32 \
      --lora_dropout 0.05 \
      --save_steps 100 \
      --logging_steps 10

END=`date`
endtime=$(date +%s)
echo "Job start at" $START
echo "Job end   at" $END