#!/bin/bash

#SBATCH --job-name=qwen3_base    # Name of your job
#SBATCH --output=/home/rberger/rlvr/GRPO_prompt_replay/outputs/reports/baseline_qwen3_dolci_%j.out  # Standard output and error log (%j expands to jobId0
#SBATCH --partition=gpu_h100           # Partition (queue)
#SBATCH --gres=gpu:h100:4              # Request N H100 GPU
#SBATCH --time=10:00:00                # Time limit (HH:MM:SS)
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=4            # Number of tasks (processes) per node


#############################################
# ================== SETUP ==================
#############################################

echo "Initial working directory: $(pwd)"

module purge

module load 2024
module load CUDA/12.6.0

export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"


cd /home/rberger/rlvr


export UV_LINK_MODE=${UV_LINK_MODE:-copy}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/scratch-shared/rberber/cache/uv}
export UV_NO_DEV=${UV_NO_DEV:-1}
export RAY_DASHBOARD_ENABLED=${RAY_DASHBOARD_ENABLED:-0}


# Tell uv to use the existing venv instead of creating a new one in Ray workers
# This prevents version mismatches when Ray packages the project
# Use realpath to handle symlinks (e.g., /home/rberger -> /gpfs/home6/rberger)
export UV_PROJECT_ENVIRONMENT="$(realpath /home/rberger/rlvr/GRPO_prompt_replay/open-instruct/.venv)"


cd /home/rberger/rlvr

# Auth (ensure these are set in your environment before running) from .env file
set -a
source .env
set +a

cd /home/rberger/rlvr/GRPO_prompt_replay/open-instruct


# Ensure the local editable dependency exists for uv sync.
if [[ ! -d vllm_olmo2.5 ]]; then
    echo "[info] Cloning custom vLLM fork into vllm_olmo2.5..."
    git clone -b shanea/olmo2-retrofit https://github.com/2015aroras/vllm.git vllm_olmo2.5
fi

pip install uv

uv sync

uv pip install torch

# Environment (adjust as needed for your cluster session)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1

export SCRATCH_CACHE_ROOT=${SCRATCH_CACHE_ROOT:-/scratch-shared/rberber/cache}
mkdir -p "${SCRATCH_CACHE_ROOT}"
export HF_HOME="${HF_HOME:-${SCRATCH_CACHE_ROOT}/hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${SCRATCH_CACHE_ROOT}/transformers_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_CACHE_ROOT}/hf_datasets_cache}"
export RAY_TMPDIR="${RAY_TMPDIR:-${SCRATCH_CACHE_ROOT}/ray_tmp}"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH_CACHE_ROOT}/torch_home}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$RAY_TMPDIR" "$TORCH_HOME"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:256"}
# Single-node NCCL usually doesn't need IB; disable to avoid hiccups
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
# Mirrors program default but set here too
export NCCL_CUMEM_ENABLE=0

cd /home/rberger/rlvr/GRPO_prompt_replay/open-instruct

export RAY_ADDRESS=local



#############################################
# ================== RUNCONFIG ==================
#############################################

export MODEL_NAME_OR_PATH="Qwen/Qwen3-1.7B-Base"
export GS_MODEL_NAME="qwen3-1.7b-base"


#DATASETS="${DATASETS:-ai2-adapt-dev/rlvr_open_reasoner_math 4000}"

#DATASETS="saurabh5/DAPO-Math-17k-Processed_filtered_olmo_completions_new_template_filtered 1500 saurabh5/MATH_3000_Filtered_olmo_completions_new_template_filtered 500"

#DATASETS="saurabh5/DAPO-Math-17k-Processed_filtered_olmo_completions_new_template_filtered 10000"


DATASETS="allenai/Dolci-RL-Zero-Math-7B 1.0"
#DATASETS="ai2-adapt-dev/deepscaler-gt 1.0"

export DATASETS="${DATASETS}"

### WHATOUT WERE NOT USING NOW THE LOCAL EVAL SAMPLE COUNT


export EVALS="${EVALS:-aime:zs_cot_r1::pass_at_32_2024_dapo,aime:zs_cot_r1::pass_at_32_2025_dapo}"
export LOCAL_EVAL_SAMPLE_COUNT="${LOCAL_EVAL_SAMPLE_COUNT:256}" # FOR TESTING USE ONLY 16, 0 to disable

# Prompt replay settings
export ENABLE_PROMPT_REPLAY="${ENABLE_PROMPT_REPLAY:-True}"
export PROMPT_REPLAY_FRACTION="${PROMPT_REPLAY_FRACTION:-0.9}"
export PROMPT_REPLAY_COOLDOWN_STEPS="${PROMPT_REPLAY_COOLDOWN_STEPS:-2}"
export PROMPT_REPLAY_MAX_REUSE_TIME="${PROMPT_REPLAY_MAX_REUSE_TIME:-15}"
export PROMPT_REPLAY_MIN_PASS_RATE="${PROMPT_REPLAY_MIN_PASS_RATE:-0.01}"
export PROMPT_REPLAY_MAX_PASS_RATE="${PROMPT_REPLAY_MAX_PASS_RATE:-0.9}"
export PX_DEPENDENT_COOLDOWN_STEPS="${PX_DEPENDENT_COOLDOWN_STEPS:-True}"
export PX_COOLDOWN_DIST_0="${PX_COOLDOWN_DIST_0:-5}"
export PX_COOLDOWN_DIST_1="${PX_COOLDOWN_DIST_1:-10}"
export PX_COOLDOWN_DIST_2="${PX_COOLDOWN_DIST_2:-15}"
export PX_COOLDOWN_DIST_3="${PX_COOLDOWN_DIST_3:-20}"


# Prompt pass curriculum settings
export ENABLE_PROMPT_PASS_CURRICULUM="${ENABLE_PROMPT_PASS_CURRICULUM:-False}"
export ZERO_PASS_CURRICULUM_FRACTION="${ZERO_PASS_CURRICULUM_FRACTION:-0.01}"
export PROMPT_PASS_CURRICULUM_05SORT="${PROMPT_PASS_CURRICULUM_05SORT:-False}"


seed=123
export SEED="${seed}"

export NO_RESAMPLING_PASS_RATE="${NO_RESAMPLING_PASS_RATE:-0.9}"

# Length settings
export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
export RESPONSE_LEN="${RESPONSE_LEN:-7168}"
export PACK_LEN="${PACK_LEN:-8192}"

# VLLM num engines
export VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-1}"



# Experiment name generation
now=$(date +%s)
hours=$(date -d @"$now" -u +%H)
minutes=$(date -d @"$now" -u +%M)
seconds=$(date -d @"$now" -u +%S)

name="promptreplay_5101520_qwen3_dolci"

export EXP_NAME="${EXP_NAME:-${name}_}"

bash /home/rberger/rlvr/GRPO_prompt_replay/open-instruct/scripts/train/olmo3/qwen_math_1_5.sh "$@"
