#!/bin/bash
#SBATCH -p gpu                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16                      # Specify number of nodes and processors per task
#SBATCH --gpus-per-node=4		# Specify total number of GPUs
#SBATCH --ntasks-per-node=4             # Specify number of tasks per node
#SBATCH -t 1:00:00                      # Specify maximum time limit (hour: minute: second)
#SBATCH -A zz991xxx                     # Specify project name
#SBATCH -J DDP_1N                       # Specify job name

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_PORT="$MASTER_PORT

export WORLD_SIZE=4   # should be obtained from $(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module load Miniforge3/25.3.0-3
conda activate pytorch-2.8.0

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn

START=`date`
starttime=$(date +%s)

srun python DDP.py --epochs=10          # Run your program or executable code

END=`date`
endtime=$(date +%s)
echo "Job start at" $START
echo "Job end   at" $END

