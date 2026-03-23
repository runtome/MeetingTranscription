#!/bin/bash
#SBATCH -p gpu                       # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16                   # Specify number of nodes and processors per task
#SBATCH --gpus-per-node=1            # Specify the number of GPUs
#SBATCH --ntasks-per-node=1          # Specify tasks per node
#SBATCH -t 100:00:00                 # Specify maximum time limit (hour: minute: second)
#SBATCH -A zz991010                  # Specify project name
#SBATCH -J Audio_ME                  # Specify job name
#SBATCH -o output.txt                # Specify output file name

module load Apptainer/1.1.6

# ==============================
# GPU Diagnostic
# ==============================
echo "=== GPU Diagnostic ==="
apptainer exec --nv -B $PWD:$PWD audio.sif python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('Device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    print('GPU memory:', round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2), 'GB')
else:
    print('WARNING: CUDA not available! Will run on CPU (very slow)')
"

# ==============================
# Check installation
# ==============================
echo "=== Check Installation ==="
apptainer exec audio.sif pip list 2>/dev/null | grep -i whisper
apptainer exec audio.sif python -c "import whisper; print('Whisper OK')"

# ==============================
# Copy data to local scratch for faster I/O
# ==============================
echo "=== Copying data to local scratch ==="
LOCAL_INPUT=$TMPDIR/test
LOCAL_SCRIPT=$TMPDIR/batch_to_csv.py
cp -r $PWD/test $LOCAL_INPUT
cp $PWD/batch_to_csv.py $LOCAL_SCRIPT
echo "Copied $(ls $LOCAL_INPUT/*.mp3 | wc -l) files to $LOCAL_INPUT"

# ==============================
# Run transcription
# ==============================
echo "=== Running Transcription ==="

apptainer exec --nv \
  -B $TMPDIR:$TMPDIR \
  -B $PWD:$PWD \
  -B $HOME/.cache:/root/.cache \
  audio.sif \
  python -u $LOCAL_SCRIPT \
  --input_dir $LOCAL_INPUT \
  --output_file $PWD/result.csv
