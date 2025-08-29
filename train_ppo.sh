# export CUDA_VISIBLE_DEVICES=0,1
export N_GPUS=8
export ROLLOUT_TP_SIZE=1
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True
# export VLLM_ATTENTION_BACKEND=XFORMERS

# All the env variables below are set to 0 by default
export WITHLENGTH=0
export REFINEDREWARD=0
export COARSEREWARD=0
export STRICTMATCH=0
export CORRECTMAX1=0
export MAX1STEP30MAX3=0
export SCHEDULEREWARD=0
export SCHEDULELENGTH=0

export CUDA_HOME=$CONDA_PREFIX
export RAY_DEDUP_LOGS=0
export PYTHONUNBUFFERED=1

export DATA_DIR="./dataset"
export BASE_MODEL="/mnt/workspace/models/xxx" # e.g., "Qwen2.5-3b-Instruct"
export EXPERIMENT_NAME="qwen2.5_vl-7b" # e.g., "grpo-qwen2.5-3b"
bash ./examples/ppo_trainer/run_ppo.sh
