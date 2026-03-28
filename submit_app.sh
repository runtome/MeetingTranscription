#!/bin/bash
#SBATCH -p gpu				# Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16          		# Specify number of nodes and processors per task
#SBATCH --gpus-per-node=1               # Specify the number of GPUs
#SBATCH --ntasks-per-node=1		# Specify tasks per node
#SBATCH -t 100:00:00			# Specify maximum time limit (hour: minute: second)
#SBATCH -A zz991010			# Specify project name
#SBATCH -J Audio_ME		# Specify job name
#SBATCH -o output.txt		# Specify output file name with job ID

module load Apptainer/1.1.6 # Load the Apptainer module
# apptainer exec --nv -B $PWD:$PWD audio.sif python batch_to_csv.py # Run your program

module load Apptainer/1.1.6



# ==============================
# Check installation
# ==============================
echo "Checking installed packages..."
apptainer exec audio.sif pip list | grep whisper

echo "Testing whisper import..."
apptainer exec audio.sif python -c "import whisper; print('Whisper OK')"

# ==============================
# Run transcription
# ==============================
echo "Running transcription..."

apptainer exec --nv \
-B $PWD:$PWD \
audio.sif \
python -u batch_to_csv.py \
--input_dir test \
--output_file result.csv
