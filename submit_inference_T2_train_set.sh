#!/bin/bash
#SBATCH -p gpu				# Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16          		# Specify number of nodes and processors per task
#SBATCH --gpus-per-node=1               # Specify the number of GPUs
#SBATCH --ntasks-per-node=1		# Specify tasks per node
#SBATCH -t 50:00:00			# Specify maximum time limit (hour: minute: second)
#SBATCH -A zz991010			# Specify project name
#SBATCH -J Inf_T2T		# Specify job name
#SBATCH -o output_inferance_T2TrainSet.txt		# Specify output file name with job ID

module load Miniforge3/25.3.0-3
conda activate /project/zz991000-zdeva/zz991010/env

export HF_HOME=/project/zz991000-zdeva/zz991010/hf/misc
export HF_DATASETS_CACHE=/project/zz991000-zdeva/zz991010/hf/datasets
export TRANSFORMERS_CACHE=/project/zz991000-zdeva/zz991010/hf/models
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

START=`date`
starttime=$(date +%s)

srun python inferance_original.py \
  --model ./whisper-thai-finetuned \
  --test_dir ./train/audio \
  --output ./train/inferance_T2.csv

END=`date`
endtime=$(date +%s)
echo "Job start at" $START
echo "Job end   at" $END