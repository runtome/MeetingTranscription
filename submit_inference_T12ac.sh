#!/bin/bash
#SBATCH -p gpu				# Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16          		# Specify number of nodes and processors per task
#SBATCH --gpus-per-node=1               # Specify the number of GPUs
#SBATCH --ntasks-per-node=1		# Specify tasks per node
#SBATCH -t 24:00:00			# Specify maximum time limit (hour: minute: second)
#SBATCH -A zz991010			# Specify project name
#SBATCH -J In_T12ac		# Specify job name
#SBATCH -o output_inferance_T12ac.txt		# Specify output file name with job ID

module load Miniforge3/25.3.0-3
conda activate /project/zz991000-zdeva/zz991010/env

export HF_HOME=/project/zz991000-zdeva/zz991010/hf/misc
export HF_DATASETS_CACHE=/project/zz991000-zdeva/zz991010/hf/datasets
export TRANSFORMERS_CACHE=/project/zz991000-zdeva/zz991010/hf/models
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

START=`date`
starttime=$(date +%s)

srun  python inference_whisper_hf.py \
    --model ./whisper-thai-finetuned-t12a \
    --output results/results_T12a_clean.csv --clean

END=`date`
endtime=$(date +%s)
echo "Job start at" $START
echo "Job end   at" $END