#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPUS=8
# export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export ROLLOUT_TP_SIZE=1
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True
export VLLM_USE_V1=1
export VLLM_FLASH_ATTN_VERSION=3
export VLLM_ENABLE_V1_MULTIPROCESSING=1

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
export BASE_MODEL="/mnt/workspace/models/Qwen3-VL-8B-Thinking" # e.g., "Qwen2.5-3b-Instruct"
export EXPERIMENT_NAME="qwen3_vl-8b" # e.g., "grpo-qwen2.5-3b"
bash ./examples/grpo_trainer/run_grpo.sh
