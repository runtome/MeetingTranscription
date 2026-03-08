#!/bin/bash
#SBATCH -p gpu                    # GPU partition
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH -N 1                      # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 06:00:00               # Max 6 hours
#SBATCH -J batch_transcribe
#SBATCH -o batch_transcribe_%j.out
#SBATCH -e batch_transcribe_%j.err

module load apptainer

# Run with GPU access, bind test/ directory and output location
apptainer exec --nv \
    --bind ./test:/workspace/test \
    --bind ./output:/workspace/output \
    batch_to_csv.sif \
    python batch_to_csv.py \
        --input_dir test \
        --output_file output/submission.csv \
        --model_size medium
