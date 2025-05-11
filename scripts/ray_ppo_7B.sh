#!/bin/bash

#SBATCH --job-name=zj_ppo_7B_8_2     # Job name
#SBATCH --partition=browse        # Specify Slurm partition
#SBATCH --account=browse
#SBATCH --gres=gpu:8              # Request 4 GPUs
#SBATCH --nodes=1                 # Request 1 compute node
#SBATCH --ntasks-per-node=1       # Run 1 task (Jupyter Lab) on the node
#SBATCH --cpus-per-task=224         # CPUs for Jupyter (adjust as needed)
#SBATCH --mem=1600G                 # Memory for Jupyter (adjust as needed)
#SBATCH --time=72:00:00           # Max job runtime (HH:MM:SS), e.g., 4 hours
#SBATCH --output="/project/browse/zhaojian/task/bin/%j.out" # %j is the Job ID
#SBATCH --error="/project/browse/zhaojian/task/%j.err"

HOME=/project/browse/zhaojian
PROJECT_ROOT=$HOME/simpleRL-reason
cd $PROJECT_ROOT

# --- start Ray ---
RAY_HEAD_PORT=${RAY_HEAD_PORT:-6379}

HEAD_NODE_HOSTNAME=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
HEAD_NODE_IP=$(getent hosts $HEAD_NODE_HOSTNAME | awk '{ print $1 }')

if [ -z "$HEAD_NODE_IP" ]; then
    echo "Fail to get ($HEAD_NODE_HOSTNAME) IP."
    echo "Please check other methods to get the IP."
    exit 1
fi

echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Ray Head Node Hostname: $HEAD_NODE_HOSTNAME"
echo "Ray Head Node IP: $HEAD_NODE_IP"
echo "Ray Head Port: $RAY_HEAD_PORT"
echo "Current Node Hostname: $(hostname)"
echo "Current Node ID (SLURM_NODEID): $SLURM_NODEID"
echo "Current Global Process ID (SLURM_PROCID): $SLURM_PROCID"

if [ "$SLURM_NODEID" -eq 0 ]; then
    ray start --head \
        --node-ip-address="$HEAD_NODE_IP" \
        --port="$RAY_HEAD_PORT" \
        --num-gpus=8 \
        --block &
    
    bash scripts/train_ppo_math_tune_ray.sh \
        --model_name Qwen2.5-1.5B \
        --dataset_name simplelr_abel_level1to4 \
        --max_response_length 8192  \
        --train_batch_size 1024 \
        --rollout_n 8 \
        --kl_loss_coef 0.0001 \
        --entropy_coeffient 0.001 \
        --rollout_gpu_memory_util 0.4 \
        --rollout_tp 2 \
        --save_freq 5
    
    TRAINING_EXIT_CODE=$?
    if [ ${TRAINING_EXIT_CODE} -ne 0 ]; then
        echo "ERROR: Python training script exited with non-zero code ${TRAINING_EXIT_CODE}."
        exit ${TRAINING_EXIT_CODE}
    fi
    
    echo "Python training script finished successfully."
    exit 0
else
    ray start \
        --address="$HEAD_NODE_IP:$RAY_HEAD_PORT" \
        --num-gpus=8 \
        --block &
    
    RAY_START_PID=$!
    echo "Ray Worker PID: ${RAY_START_PID}"
    echo "Ray Worker started. Keeping pod alive..."
    
    wait ${RAY_START_PID} # Wait for Ray start process to exit (usually when killed)
    exit 0
fi

