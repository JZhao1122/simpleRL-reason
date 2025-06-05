#!/bin/bash

source $HOME/.bashrc

NEW_HOME=/cpfs02/user/liurunze
eval "$(${NEW_HOME}/miniforge3/bin/conda shell.bash hook)"

export conda_name=zj_or1
conda activate $conda_name

export PYTHONPATH=${NEW_HOME}/_/simpleRL-reason
export PYTHON_EXECUTABLE=$(which python)
cd ${PYTHONPATH}

model_path=${NEW_HOME}/hf_models

name=$1
if [ -z "$name" ]; then
    echo "Error: Model name argument is required."
    exit 1
fi
VALUE_MODEL_PATH=${model_path}/${name}

NUM_RM_WORKER=${NUM_RM_WORKER:-1}
HOST_ADDR=${HOST_ADDR:-"0.0.0.0"}
CONTROLLER_PORT=${CONTROLLER_PORT:-"10014"}
WORKER_BASE_PORT=${WORKER_BASE_PORT:-"10081"}

LOGDIR_BASE=${PYTHONPATH}/_outputs/logs_fastchat/log_fc
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export LOGDIR="${LOGDIR_BASE}_${TIMESTAMP}"
mkdir -p "$LOGDIR"
controller_log_file="${LOGDIR}/controller.log"
worker_log_dir="${LOGDIR}/workers"
mkdir -p "$worker_log_dir"

controller_session_name="fastchat_controller_main"
echo "Attempting to start FastChat Controller..."

if tmux has-session -t $controller_session_name 2>/dev/null; then
    echo "Controller tmux session $controller_session_name already exists. Killing it."
    tmux kill-session -t $controller_session_name
fi
tmux start-server

tmux new-session -s $controller_session_name -d \
    "bash -c 'source \$HOME/.bashrc && eval \"\$(${NEW_HOME}/miniforge3/bin/conda shell.bash hook)\" && conda activate ${conda_name} && export LOGDIR=\"${LOGDIR}\" && cd \"${PYTHONPATH}\" && \"${PYTHON_EXECUTABLE}\" -m fastchat.serve.controller --port \"${CONTROLLER_PORT}\" --host \"${HOST_ADDR}\" &> \"${controller_log_file}\"'"

echo "FastChat Controller started in background. Logs: ${controller_log_file}"
echo "Waiting for controller to initialize (5 seconds)..."
sleep 5

echo "Starting workers in background..."
for i in $(seq 0 $((NUM_RM_WORKER-1)))
do
    WORKER_PORT=$((WORKER_BASE_PORT+i))
    worker_log_file="${worker_log_dir}/reward_worker_${i}_port${WORKER_PORT}.log"

    setup_commands="source \$HOME/.bashrc && eval \"\$(${NEW_HOME}/miniforge3/bin/conda shell.bash hook)\" && conda activate ${conda_name} && export LOGDIR=\"${LOGDIR}\" && cd \"${PYTHONPATH}\""

    if [[ "$VALUE_MODEL_PATH" =~ "dummy" ]]; then
        worker_exec_command="echo \"Dummy worker ${i} on port ${WORKER_PORT} active. Logging to ${worker_log_file}\" && sleep 3600" # dummy 命令也输出日志提示
    else
        worker_exec_command="\"${PYTHON_EXECUTABLE}\" -m evaluation.infer_module.llm_service.workers.reward_model_worker --num-gpus 1 --model-path \"${VALUE_MODEL_PATH}\" --controller-address \"http://${HOST_ADDR}:${CONTROLLER_PORT}\" --host \"${HOST_ADDR}\" --port \"${WORKER_PORT}\" --worker-address \"http://${HOST_ADDR}:${WORKER_PORT}\""
    fi
    
    echo "Starting worker ${i} on port ${WORKER_PORT}. Logs: ${worker_log_file}"
    
    (
        bash -c "${setup_commands} && ${worker_exec_command}" &> "${worker_log_file}"
    ) &

    sleep 0.2
done

echo "Waiting for vllm server..."
tail -f "${worker_log_file}" | grep -q "Application startup complete."

echo "Script finished launching background processes."
echo "To stop controller: tmux kill-session -t ${controller_session_name}"
echo "To stop workers: pkill -f 'reward_model_worker' (or use PIDs if saved)"
