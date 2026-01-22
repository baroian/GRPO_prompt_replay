#!/bin/bash

# Snellius single-node launch script for the GRPO math run using
# Qwen2.5-Math-1.5B. Designed for interactive use (e.g., `srun bash`).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_INSTRUCT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_ROOT="$(cd "${OPEN_INSTRUCT_ROOT}/.." && pwd)"

MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-1.5B"
GS_MODEL_NAME="qwen25_math_1_5b"
#MODEL_NAME_OR_PATH="allenai/Olmo-3-1025-7B"
#GS_MODEL_NAME="olmo3-1025-7b"

# Qwen2.5 Math 1.5B supports ~4k context; cap lengths accordingly.
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
RESPONSE_LEN="${RESPONSE_LEN:-2048}"
PACK_LEN="${PACK_LEN:-3072}"

# Dataset and evaluation settings (set via SLURM or environment)
# Defaults are set in run_test.slurm, but can be overridden here for direct execution
DATASETS="${DATASETS:-ai2-adapt-dev/rlvr_open_reasoner_math 2000}"
#LOCAL_EVALS="${LOCAL_EVALS:-mnoukhov/aime2024-25-rlvr 1.0 mnoukhov/aime2024-25-rlvr 1.0}"
#LOCAL_EVAL_SPLITS="${LOCAL_EVAL_SPLITS:-test_2024 test_2024 test_2025 test_2025}"
EVALS="${EVALS:-aime:zs_cot_r1::pass_at_32_2024_dapo,aime:zs_cot_r1::pass_at_32_2025_dapo}"

# Local eval subset: quick feedback from training data
LOCAL_EVAL_SAMPLE_COUNT="${LOCAL_EVAL_SAMPLE_COUNT:-128}"
LOCAL_EVAL_TIMEOUT="${LOCAL_EVAL_TIMEOUT:-300}"

# Benchmark evaluation: separate full benchmark (e.g., MATH-500)
# This logs to wandb under "benchmark/" prefix, separate from "eval/"
# 
# MULTI-BENCHMARK EVALUATION:
# Use auto_convert_benchmark_format to evaluate on multiple benchmarks with different column formats!
# Format: "dataset_name num_samples dataset_name num_samples ..."
# Splits: "split split ..." (one per dataset)
#
# Supported benchmarks (auto-detected by auto_convert_benchmark_format):
#   - HuggingFaceH4/MATH-500: problem/answer columns, split=test
#   - math-ai/minervamath: question/answer columns, split=test, 272 total examples
#   - Hothan/OlympiadBench:OE_TO_maths_en_COMP: question/final_answer columns, split=train (no test!), 674 examples
#   - AI-MO/aimo-validation-aime: problem/answer columns, split=train
#   - AI-MO/aimo-validation-amc: problem/answer columns, split=train
#
# Example multi-benchmark config:
#   BENCHMARK_EVALS="HuggingFaceH4/MATH-500 128 math-ai/minervamath 272"
#   BENCHMARK_EVAL_SPLITS="test test"
#   BENCHMARK_TRANSFORM_FN="auto_convert_benchmark_format rlvr_tokenize_v1 rlvr_max_length_filter_v1"
#


BENCHMARK_EVALS="${BENCHMARK_EVALS:-HuggingFaceH4/MATH-500 100 math-ai/minervamath 100 Hothan/OlympiadBench:OE_TO_maths_en_COMP 100}"
BENCHMARK_EVAL_SPLITS="${BENCHMARK_EVAL_SPLITS:-test test train}"
#BENCHMARK_EVALS="HuggingFaceH4/MATH-500 128 math-ai/minervamath 272 Hothan/OlympiadBench:OE_TO_maths_en_COMP 674" \
#BENCHMARK_EVAL_SPLITS="test test train" \
BENCHMARK_EVAL_EVERY="${BENCHMARK_EVAL_EVERY:-}"  # Empty means use local_eval_every
# Transform functions for benchmark datasets (converts non-standard formats to RLVR format)
# RECOMMENDED: Use auto_convert_benchmark_format for multi-benchmark evaluation
# This auto-detects column formats (problem/question, answer/final_answer) and handles them correctly
BENCHMARK_TRANSFORM_FN="${BENCHMARK_TRANSFORM_FN:-auto_convert_benchmark_format rlvr_tokenize_v1 rlvr_max_length_filter_v1}"


# Prompt replay settings
ENABLE_PROMPT_REPLAY="${ENABLE_PROMPT_REPLAY:-False}"
PROMPT_REPLAY_FRACTION="${PROMPT_REPLAY_FRACTION:-0.5}"
PROMPT_REPLAY_COOLDOWN_STEPS="${PROMPT_REPLAY_COOLDOWN_STEPS:-5}"
PROMPT_REPLAY_MAX_REUSE_TIME="${PROMPT_REPLAY_MAX_REUSE_TIME:-5}"
PROMPT_REPLAY_MIN_PASS_RATE="${PROMPT_REPLAY_MIN_PASS_RATE:-0.24}"
PROMPT_REPLAY_MAX_PASS_RATE="${PROMPT_REPLAY_MAX_PASS_RATE:-0.7}"

# Prompt pass curriculum settings
ENABLE_PROMPT_PASS_CURRICULUM="${ENABLE_PROMPT_PASS_CURRICULUM:-False}"
ZERO_PASS_CURRICULUM_FRACTION="${ZERO_PASS_CURRICULUM_FRACTION:-0.25}"
PROMPT_PASS_CURRICULUM_05SORT="${PROMPT_PASS_CURRICULUM_05SORT:-False}"

# when to discard a prompt from the dataset
NO_RESAMPLING_PASS_RATE="${NO_RESAMPLING_PASS_RATE:-0.85}"

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch-shared/rberber/rlvr}"
mkdir -p "${SCRATCH_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_ROOT}/outputs/qwen25-math-rlzero-math/checkpoints}"
mkdir -p "${OUTPUT_DIR}"

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_STATE_DIR="${OUTPUT_DIR}/state_${RUN_ID}"

NODE_SETUP="${OPEN_INSTRUCT_ROOT}/node_setup.sh"

seed=${SEED:-1}

# shellcheck disable=SC1090
source "${NODE_SETUP}"

cd "${OPEN_INSTRUCT_ROOT}"

echo "[info] Using repo root: ${REPO_ROOT}"
echo "[info] Output dir: ${OUTPUT_DIR}"
echo "[info] Local eval subset sample count: ${LOCAL_EVAL_SAMPLE_COUNT}"
echo "[info] Benchmark dataset: ${BENCHMARK_EVALS}"
echo "[info] Benchmark splits: ${BENCHMARK_EVAL_SPLITS}"
echo "[info] Benchmark transform: ${BENCHMARK_TRANSFORM_FN}"

# Experiment name (set via SLURM or generate here for direct execution)
if [[ -z "${EXP_NAME:-}" ]]; then
    now=$(date +%s)
    hours=$(date -d @"$now" -u +%H)
    minutes=$(date -d @"$now" -u +%M)
    seconds=$(date -d @"$now" -u +%S)
    EXP_NAME="baseline_deepscaler_promptreplay_${hours}h${minutes}m${seconds}s"
fi

#NOTE IN FLIGHTS ARE TERRIBLY SLOW - nevermind, they work great
#    --inflight_updates \


# shellcheck disable=SC2086
uv run python open_instruct/grpo_fast.py \
    --exp_name "${EXP_NAME}" \
    --beta 0.0 \
    --async_steps 2 \
    --inflight_updates \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --advantage_normalization_type centered \
    --active_sampling \
    --no_resampling_pass_rate $NO_RESAMPLING_PASS_RATE \
    --num_samples_per_prompt_rollout 16 \
    --num_unique_prompts_rollout 32 \
    --num_mini_batches 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --kl_estimator kl3 \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits train \
    --local_eval_subset_sample_count "${LOCAL_EVAL_SAMPLE_COUNT}" \
    --local_eval_timeout "${LOCAL_EVAL_TIMEOUT}" \
    --dataset_mixer_benchmark_list $BENCHMARK_EVALS \
    --dataset_mixer_benchmark_list_splits $BENCHMARK_EVAL_SPLITS \
    --dataset_transform_fn_benchmark $BENCHMARK_TRANSFORM_FN \
    ${BENCHMARK_EVAL_EVERY:+--benchmark_eval_every "${BENCHMARK_EVAL_EVERY}"} \
    --max_prompt_token_length "${MAX_PROMPT_LEN}" \
    --response_length "${RESPONSE_LEN}" \
    --pack_length "${PACK_LEN}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --chat_template_name olmo_thinker_dapo \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 512000 \
    --deepspeed_stage 2 \
    --num_learners_per_node 2 \
    --vllm_num_engines 2\
    --vllm_tensor_parallel_size 1 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --record_entropy true \
    --seed ${seed} \
    --local_eval_every 20 \
    --save_freq 5000 \
    --checkpoint_state_freq 5000 \
    --checkpoint_state_dir "${CHECKPOINT_STATE_DIR}" \
    --gradient_checkpointing \
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --mask_truncated_completions True \
    --with_tracking \
    --wandb_project_name "olmo3_curriculum" \
    --eval_on_step_0 True \
    --output_dir "${OUTPUT_DIR}" \
    --prompt_pass_table_dir "${OPEN_INSTRUCT_ROOT}/UC" \
    --enable_prompt_replay "${ENABLE_PROMPT_REPLAY}" \
    --prompt_replay_fraction "${PROMPT_REPLAY_FRACTION}" \
    --prompt_replay_cooldown_steps "${PROMPT_REPLAY_COOLDOWN_STEPS}" \
    --prompt_replay_max_reuse_time "${PROMPT_REPLAY_MAX_REUSE_TIME}" \
    --prompt_replay_min_pass_rate "${PROMPT_REPLAY_MIN_PASS_RATE}" \
    --prompt_replay_max_pass_rate "${PROMPT_REPLAY_MAX_PASS_RATE}" \
    --enable_prompt_pass_curriculum "${ENABLE_PROMPT_PASS_CURRICULUM}" \
    --zero_pass_curriculum_fraction "${ZERO_PASS_CURRICULUM_FRACTION}" \
    --prompt_pass_curriculum_05sort "${PROMPT_PASS_CURRICULUM_05SORT}" \
    "$@"


#    --enable_prompt_replay True \
    #--prompt_replay_fraction 0.5 \
    #--prompt_replay_cooldown_steps 5 \
#--with_tracking \

