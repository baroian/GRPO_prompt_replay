#!/bin/bash

#SBATCH --job-name=base    # Name of your job
#SBATCH --output=/home/rberger/rlvr/GRPO_prompt_replay/outputs/reports/base_%j.out  # Standard output and error log (%j expands to jobId0
#SBATCH --partition=gpu_h100           # Partition (queue)
#SBATCH --gres=gpu:h100:4              # Request N H100 GPU
#SBATCH --time=10:00:00                # Time limit (HH:MM:SS)
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=4            # Number of tasks (processes) per node

echo "Initial working directory: $(pwd)"

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


. GRPO_prompt_replay/open-instruct/node_setup.sh



#DATASETS="${DATASETS:-ai2-adapt-dev/rlvr_open_reasoner_math 4000}"

#DATASETS="saurabh5/DAPO-Math-17k-Processed_filtered_olmo_completions_new_template_filtered 1500 saurabh5/MATH_3000_Filtered_olmo_completions_new_template_filtered 500"

#DATASETS="saurabh5/DAPO-Math-17k-Processed_filtered_olmo_completions_new_template_filtered 10000"

DATASETS="allenai/Dolci-RL-Zero-Math-7B 1.0"
#DATASETS="ai2-adapt-dev/deepscaler-gt 1.0"


export DATASETS="${DATASETS}"

### WHATOUT WERE NOT USING NOW THE LOCAL EVAL SAMPLE COUNT


export EVALS="${EVALS:-aime:zs_cot_r1::pass_at_32_2024_dapo,aime:zs_cot_r1::pass_at_32_2025_dapo}"
export LOCAL_EVAL_SAMPLE_COUNT="${LOCAL_EVAL_SAMPLE_COUNT:256}" # FOR TESTING USE ONLY 16

# Prompt replay settings
export ENABLE_PROMPT_REPLAY="${ENABLE_PROMPT_REPLAY:-False}"
export PROMPT_REPLAY_FRACTION="${PROMPT_REPLAY_FRACTION:-0.9}"
export PROMPT_REPLAY_COOLDOWN_STEPS="${PROMPT_REPLAY_COOLDOWN_STEPS:-5}"
export PROMPT_REPLAY_MAX_REUSE_TIME="${PROMPT_REPLAY_MAX_REUSE_TIME:-30}"
export PROMPT_REPLAY_MIN_PASS_RATE="${PROMPT_REPLAY_MIN_PASS_RATE:-0.01}"
export PROMPT_REPLAY_MAX_PASS_RATE="${PROMPT_REPLAY_MAX_PASS_RATE:-0.85}"

# Prompt pass curriculum settings
export ENABLE_PROMPT_PASS_CURRICULUM="${ENABLE_PROMPT_PASS_CURRICULUM:-False}"
export ZERO_PASS_CURRICULUM_FRACTION="${ZERO_PASS_CURRICULUM_FRACTION:-0.25}"
export PROMPT_PASS_CURRICULUM_05SORT="${PROMPT_PASS_CURRICULUM_05SORT:-True}"


seed=123
export SEED="${seed}"

export NO_RESAMPLING_PASS_RATE="${NO_RESAMPLING_PASS_RATE:-0.9}"

# Length settings
export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
export RESPONSE_LEN="${RESPONSE_LEN:-7684}"
export PACK_LEN="${PACK_LEN:-8196}"

# VLLM num engines
export VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-1}"



# Experiment name generation
now=$(date +%s)
hours=$(date -d @"$now" -u +%H)
minutes=$(date -d @"$now" -u +%M)
seconds=$(date -d @"$now" -u +%S)

name="base_r1_distill_qwen"

export EXP_NAME="${EXP_NAME:-${name}_}"

bash /home/rberger/rlvr/GRPO_prompt_replay/open-instruct/scripts/train/olmo3/qwen_math_1_5.sh "$@"
