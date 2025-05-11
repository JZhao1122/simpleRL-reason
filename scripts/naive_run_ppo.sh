#! /bin/bash

HOME=/project/browse/zhaojian
PROJECT_ROOT=$HOME/simpleRL-reason
cd $PROJECT_ROOT

bash scripts/train_ppo_math_tune_ray.sh \
	--model_name Qwen2.5-0.5B \
	--dataset_name simplelr_abel_level1to4 \
	--max_response_length 8192  \
	--train_batch_size 1024 \
	--rollout_n 8 \
	--kl_loss_coef 0.0001 \
	--entropy_coeffient 0.001 \
	--rollout_gpu_memory_util 0.4 \
	--rollout_tp 2 \
	--save_freq 5