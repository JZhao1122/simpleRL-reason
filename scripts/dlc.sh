#!/bin/bash

source $HOME/.bashrc
NEW_HOME=/cpfs02/user/liurunze # 确保这是 DLC 容器能访问到的共享路径

# --- 参数解析 ---
# --- Verl 训练执行脚本的路径 ---
RUN_SCRIPT_NAME=$1 # 假设执行脚本名称
RUN_SCRIPT_PATH="${NEW_HOME}/_/simpleRL-reason/scripts/${RUN_SCRIPT_NAME}.sh" # 确保路径正确
# 接收总 GPU 数量作为第一个命令行参数
N_GPUS_PER_WORKER_NODE=$2
NNODES=$3
shift 3
# TRAIN_SCRIPT_OVERRIDES=("$@")
# --- 参数解析结束 ---

# --- DLC 和资源配置 ---
data_sources="d-x135sqzws1argjld1r"
resource_id=quota1bhq0p32wuc  # llmit6
workspace_id=84885
priority=9
#oversold_type=ForbiddenQuotaOverSold  # 如果不指定，则不使用闲时资源
oversold_type=ForceQuotaOverSold  # 如果设置为1，则只使用闲时资源
#oversold_type=AcceptQuotaOverSold  # 如果设置为2，则可接受使用闲时资源

TOTAL_GPU_COUNT=$((N_GPUS_PER_WORKER_NODE * NNODES))
echo "Requesting $NNODES worker nodes with $N_GPUS_PER_WORKER_NODE GPUs each (Total GPUs: $TOTAL_GPU_COUNT)"

# --- Worker 容器镜像和资源 ---
worker_image=pjlab-wulan-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pjlab-eflops/liurunze:liurunze-sllm02
WORKER_CPU=$((N_GPUS_PER_WORKER_NODE * 12))
WORKER_MEMORY="$((N_GPUS_PER_WORKER_NODE * 200))Gi"
WORKER_SHARED_MEMORY="$((N_GPUS_PER_WORKER_NODE * 200))Gi"

# --- 生成实验名称 ---
EXPERIMENT_NAME=${RUN_SCRIPT_NAME}


echo "Submitting training job:"
echo "  Total GPUs: ${TOTAL_GPU_COUNT}"
echo "  Nodes: ${NNODES}, GPUs per node: ${N_GPUS_PER_WORKER_NODE}"
echo "  Experiment Name: ${EXPERIMENT_NAME}"
echo "  Worker Image: ${worker_image}"
echo "  Run Script: ${RUN_SCRIPT_PATH}"
echo "============================================="

# 使用 dlc submit 提交任务
dlc submit pytorchjob \
    --name="zj_train_${EXPERIMENT_NAME}" \
    --command="bash ${RUN_SCRIPT_PATH} ${N_GPUS_PER_WORKER_NODE} ${NNODES}" \
    --data_sources=$data_sources \
    --resource_id=$resource_id \
    --workspace_id=$workspace_id \
    --priority=$priority \
    --driver=535.54.03 \
    --job_max_running_time_minutes=0 \
    --workers=$NNODES \
    --worker_image=$worker_image \
    --worker_cpu=$WORKER_CPU \
    --worker_memory=$WORKER_MEMORY \
    --worker_shared_memory=$WORKER_SHARED_MEMORY \
    --worker_gpu=$N_GPUS_PER_WORKER_NODE \
    --oversold_type=$oversold_type

# worker_image=dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/ray:2.39.0-gpu-py312-cu118-ubuntu22.04
