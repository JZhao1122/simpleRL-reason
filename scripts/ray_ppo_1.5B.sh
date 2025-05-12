#!/bin/bash

# Initialize environment
source $HOME/.bashrc
NEW_HOME=/cpfs02/user/liurunze
eval "$(${NEW_HOME}/miniforge3/bin/conda shell.bash hook)"
which conda
conda activate zj_simrl
which python

model_path=${NEW_HOME}/hf_models
data_path=${NEW_HOME}/hf_models/datasets--hkust-nlp--SimpleRL-Zoo-Data
output_path=${NEW_HOME}/_/simpleRL-reason/_outputs

export WANDB_API_KEY=b97cb56d9b9da4a7908aedcc2ca7dcde8a80643e
#================================================================================>

PROJECT_ROOT=$NEW_HOME/_/simpleRL-reason
cd $PROJECT_ROOT

export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29500}
PORT=6379
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export N_GPUS_PER_NODE=8

if [ "$RANK" -eq 0 ]; then
    echo "Starting head node (RANK=${RANK}) on port $PORT..."
    ray start --head --port=$PORT --num-gpus=$N_GPUS_PER_NODE --block &
    RAY_START_PID=$!
    echo "Ray Head PID: ${RAY_START_PID}"

    bash scripts/train_ppo_math_tune_ray.sh \
        --model_name models--Qwen--Qwen2.5-1.5B \
        --dataset_name simplelr_abel_level1to4 \
        --max_response_length 8192  \
        --train_batch_size 1024 \
        --rollout_n 8 \
        --kl_loss_coef 0.0001 \
        --entropy_coeffient 0.001 \
        --rollout_gpu_memory_util 0.4 \
        --rollout_tp 2 \
        --save_freq 10

    TRAINING_EXIT_CODE=$?
    if [ ${TRAINING_EXIT_CODE} -ne 0 ]; then
        echo "ERROR: Python training script exited with non-zero code ${TRAINING_EXIT_CODE}."
        exit ${TRAINING_EXIT_CODE} # 以 Python 命令的退出码退出脚本
    fi
    
    echo "Python training script finished successfully."
    exit 0
else
    echo "Starting worker node (RANK=${RANK}), connecting to ${MASTER_ADDR}:${PORT}..."
    ray start --address=${MASTER_ADDR}:${PORT} --num-gpus=$N_GPUS_PER_NODE --block &
    RAY_START_PID=$!
    echo "Ray Worker PID: ${RAY_START_PID}"
    echo "Ray Worker started. Keeping pod alive..."
    
    wait ${RAY_START_PID} # Wait for Ray start process to exit (usually when killed)
    exit 0
fi
