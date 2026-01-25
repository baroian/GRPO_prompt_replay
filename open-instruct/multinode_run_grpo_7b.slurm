#!/bin/bash

#SBATCH --job-name=olmo    # Name of your job
#SBATCH --output=/home/rberger/rlvr/GRPO_prompt_replay/outputs/olmo/grpo_7b_mn_%j.out  # Standard output and error log (%j expands to jobId)
#SBATCH --partition=gpu_h100           # Partition (queue)
#SBATCH --gres=gpu:h100:4              # Request N H100 GPU per node
#SBATCH --time=24:00:00                # Time limit (HH:MM:SS)
#SBATCH --nodes=3                      # Number of nodes
#SBATCH --ntasks-per-node=1            # One Ray daemon per node

set -euo pipefail

name="qwen_7b_dolci"

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
VENV_DIR="$(realpath /home/rberger/rlvr/GRPO_prompt_replay/open-instruct/.venv)"
export PATH="${VENV_DIR}/bin:${PATH}"
RAY_CMD="${VENV_DIR}/bin/ray"
export RAY_PYTHON="${VENV_DIR}/bin/python"
RAY_LOG_DIR="${RAY_LOG_DIR:-/home/rberger/rlvr/GRPO_prompt_replay/outputs/reports/ray_logs_${SLURM_JOB_ID}}"
mkdir -p "${RAY_LOG_DIR}"

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
export TORCH_HOME="${TORCH_HOME:-${SCRATCH_CACHE_ROOT}/torch_home}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME"

# Ray temp/logs should be local per-node if possible.
if [[ -d /scratch-local ]]; then
    export RAY_TMPDIR="${RAY_TMPDIR:-/scratch-local/$USER/ray/${SLURM_JOB_ID}}"
else
    export RAY_TMPDIR="${RAY_TMPDIR:-${SCRATCH_CACHE_ROOT}/ray_tmp}"
fi
mkdir -p "$RAY_TMPDIR"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:256"}
# Multi-node might prefer IB; override NCCL_IB_DISABLE=0 if your fabric supports it.
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
# Mirrors program default but set here too
export NCCL_CUMEM_ENABLE=0
# More verbose Ray logs when things go wrong (disable by setting to 1)
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}

#############################################
# ================== RAY ==================
#############################################

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node="${nodes[0]}"
ray_port="${RAY_PORT:-6379}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
NUM_CPUS_PER_NODE="${NUM_CPUS_PER_NODE:-${SLURM_CPUS_ON_NODE:-64}}"

# Ray timing configuration
RAY_HEAD_WAIT_ATTEMPTS="${RAY_HEAD_WAIT_ATTEMPTS:-60}"
RAY_HEAD_WAIT_SLEEP="${RAY_HEAD_WAIT_SLEEP:-2}"
RAY_WORKER_CONNECT_ATTEMPTS="${RAY_WORKER_CONNECT_ATTEMPTS:-60}"
RAY_WORKER_CONNECT_SLEEP="${RAY_WORKER_CONNECT_SLEEP:-2}"
RAY_JOIN_MAX_ATTEMPTS="${RAY_JOIN_MAX_ATTEMPTS:-60}"
RAY_JOIN_SLEEP="${RAY_JOIN_SLEEP:-5}"

# Network interface selection
pick_iface() {
    if [[ -n "${RAY_INTERFACE:-}" ]]; then
        echo "${RAY_INTERFACE}"
        return 0
    fi
    # Prefer the non-IB fabric unless explicitly overridden.
    for iface in eno2np0 ib-bond0 vlan-pub.136; do
        if ip link show "$iface" >/dev/null 2>&1; then
            echo "$iface"
            return 0
        fi
    done
    echo ""
}

pick_ip_from_iface() {
    local iface="$1"
    if [[ -z "$iface" ]]; then
        echo ""
        return 0
    fi
    ip -4 -o addr show dev "$iface" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -n1
}

pick_ip_fallback() {
    hostname -I 2>/dev/null | tr ' ' '\n' | grep -v '^169\.254\.' | head -n1
}

RAY_INTERFACE="$(pick_iface)"
# Ensure NCCL doesn't pick link-local; allow explicit override.
if [[ -z "${NCCL_SOCKET_IFNAME:-}" && -n "${RAY_INTERFACE}" ]]; then
    export NCCL_SOCKET_IFNAME="${RAY_INTERFACE}"
fi

# Source Ray debugging utilities
source /home/rberger/rlvr/GRPO_prompt_replay/open-instruct/scripts/ray_debug.sh

# Print debug info (set RAY_DEBUG=0 to skip)
if [[ "${RAY_DEBUG:-1}" == "1" ]]; then
    print_ray_debug_info "${nodes[@]}"
fi

# Resolve head node IP
if [[ -n "${RAY_HEAD_IP:-}" ]]; then
    head_node_ip="${RAY_HEAD_IP}"
else
    head_node_ip="$(pick_ip_from_iface "${RAY_INTERFACE}")"
    if [[ -z "${head_node_ip}" ]]; then
        head_node_ip="$(pick_ip_fallback)"
    fi
fi
if [[ -z "$head_node_ip" ]]; then
    echo "[error] Failed to resolve head node IP." >&2
    exit 1
fi

echo "[ray] Head node: $head_node ($head_node_ip)"

# Cleanup function
auto_ray_stop() {
    srun --nodes=1 --ntasks=1 -w "$head_node" "${RAY_CMD}" stop --force >/dev/null 2>&1 || true
    for node in "${nodes[@]:1}"; do
        srun --nodes=1 --ntasks=1 -w "$node" "${RAY_CMD}" stop --force >/dev/null 2>&1 || true
    done
}

# Exit trap for error logging
on_exit() {
    status=$?
    if [[ $status -ne 0 ]]; then
        dump_all_ray_logs "${nodes[@]}"
    fi
}

trap on_exit EXIT

auto_ray_stop

# Start Ray head (block in background to keep raylet alive under Slurm)
srun --nodes=1 --ntasks=1 --cpus-per-task="${NUM_CPUS_PER_NODE}" --gpus-per-task="${NUM_GPUS_PER_NODE}" --exclusive -w "$head_node" \
    --output="${RAY_LOG_DIR}/ray_head_${SLURM_JOB_ID}.out" --error="${RAY_LOG_DIR}/ray_head_${SLURM_JOB_ID}.err" \
    /bin/bash -c "mkdir -p '${RAY_TMPDIR}'; \
    cvd=\"\${CUDA_VISIBLE_DEVICES:-}\"; \
    if [[ -z \"\${cvd}\" && -n \"\${SLURM_STEP_GPUS:-}\" ]]; then \
      cvd=\${SLURM_STEP_GPUS//gpu:/}; \
    fi; \
    if [[ -z \"\${cvd}\" ]]; then \
      cvd=\$(seq -s, 0 $((NUM_GPUS_PER_NODE-1))); \
    fi; \
    export CUDA_VISIBLE_DEVICES=\"\${cvd}\"; \
    ${RAY_CMD} start --head --node-ip-address='${head_node_ip}' --port='${ray_port}' --num-gpus='${NUM_GPUS_PER_NODE}' --num-cpus='${NUM_CPUS_PER_NODE}' --dashboard-host=0.0.0.0 --block" &

sleep 10

# Wait for head node to be ready
echo "[ray] Waiting for head to accept connections..."
head_ready=0
for i in $(seq 1 "${RAY_HEAD_WAIT_ATTEMPTS}"); do
    if ${RAY_CMD} status --address "${head_node_ip}:${ray_port}" >/dev/null 2>&1; then
        head_ready=1
        break
    fi
    sleep "${RAY_HEAD_WAIT_SLEEP}"
done
if [[ "${head_ready}" -ne 1 ]]; then
    echo "[ray] ERROR: head did not become ready after $((RAY_HEAD_WAIT_ATTEMPTS * RAY_HEAD_WAIT_SLEEP))s."
    dump_ray_logs "${head_node}"
    exit 1
fi

# Start Ray workers
for node in "${nodes[@]:1}"; do
    echo "[ray] Starting worker on $node"
    srun --nodes=1 --ntasks=1 --cpus-per-task="${NUM_CPUS_PER_NODE}" --gpus-per-task="${NUM_GPUS_PER_NODE}" --exclusive -w "$node" \
        --output="${RAY_LOG_DIR}/ray_worker_${node}_${SLURM_JOB_ID}.out" --error="${RAY_LOG_DIR}/ray_worker_${node}_${SLURM_JOB_ID}.err" \
        /bin/bash -c "iface='${RAY_INTERFACE}'; \
         if [[ -n \"${RAY_INTERFACE}\" ]]; then \
           node_ip=\$(ip -4 -o addr show dev \"${RAY_INTERFACE}\" 2>/dev/null | awk '{print \$4}' | cut -d/ -f1 | head -n1); \
         fi; \
         if [[ -z \"\${node_ip:-}\" ]]; then \
           node_ip=\$(hostname -I | tr ' ' '\\n' | grep -v '^169\\.254\\.' | head -n1); \
         fi; \
         for i in \$(seq 1 ${RAY_WORKER_CONNECT_ATTEMPTS}); do \
           if ${RAY_CMD} status --address '${head_node_ip}:${ray_port}' >/dev/null 2>&1; then \
             break; \
           fi; \
           sleep ${RAY_WORKER_CONNECT_SLEEP}; \
         done; \
         mkdir -p '${RAY_TMPDIR}'; \
         cvd=\"\${CUDA_VISIBLE_DEVICES:-}\"; \
         if [[ -z \"\${cvd}\" && -n \"\${SLURM_STEP_GPUS:-}\" ]]; then \
           cvd=\${SLURM_STEP_GPUS//gpu:/}; \
         fi; \
         if [[ -z \"\${cvd}\" ]]; then \
           cvd=\$(seq -s, 0 $((NUM_GPUS_PER_NODE-1))); \
         fi; \
         export CUDA_VISIBLE_DEVICES=\"\${cvd}\"; \
         ${RAY_CMD} start --address='${head_node_ip}:${ray_port}' --node-ip-address=\${node_ip} --num-gpus='${NUM_GPUS_PER_NODE}' --num-cpus='${NUM_CPUS_PER_NODE}' --dashboard-host=0.0.0.0 --block" &
    sleep 5
done

export RAY_ADDRESS="${head_node_ip}:${ray_port}"
echo "[ray] RAY_ADDRESS set to ${RAY_ADDRESS}"

# Wait for all nodes to join
expected_nodes=${#nodes[@]}
if ! wait_for_ray_cluster "${expected_nodes}" "${nodes[@]}"; then
    exit 1
fi

print_ray_status

#############################################
# ================== RUNCONFIG ==================
#############################################

name="baseline_olmo_dolci"

### OLMO config 
export MODEL_NAME_OR_PATH="allenai/Olmo-3-1025-7B"
export GS_MODEL_NAME="olmo3-1025-7b"

export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
export RESPONSE_LEN="${RESPONSE_LEN:-7168}"
export PACK_LEN="${PACK_LEN:-8192}"

### QWEN config 

#export MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B"
#export GS_MODEL_NAME="qwen25-math-7b"

#export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
#export RESPONSE_LEN="${RESPONSE_LEN:-3584}"
#export PACK_LEN="${PACK_LEN:-4096}"



DATASETS="allenai/Dolci-RL-Zero-Math-7B 1.0"
export DATASETS="${DATASETS}"

export EVALS="${EVALS:-aime:zs_cot_r1::pass_at_32_2024_dapo,aime:zs_cot_r1::pass_at_32_2025_dapo}"
export LOCAL_EVAL_SAMPLE_COUNT="${LOCAL_EVAL_SAMPLE_COUNT:-256}"
export EVAL_ON_STEP_0="${EVAL_ON_STEP_0:-True}"

# Prompt replay settings
export ENABLE_PROMPT_REPLAY="${ENABLE_PROMPT_REPLAY:-False}"
export PROMPT_REPLAY_FRACTION="${PROMPT_REPLAY_FRACTION:-0.9}"
export PROMPT_REPLAY_COOLDOWN_STEPS="${PROMPT_REPLAY_COOLDOWN_STEPS:-5}"
export PROMPT_REPLAY_MAX_REUSE_TIME="${PROMPT_REPLAY_MAX_REUSE_TIME:-15}"
export PROMPT_REPLAY_MIN_PASS_RATE="${PROMPT_REPLAY_MIN_PASS_RATE:-0.01}"
export PROMPT_REPLAY_MAX_PASS_RATE="${PROMPT_REPLAY_MAX_PASS_RATE:-0.9}"

# Prompt pass curriculum settings
export ENABLE_PROMPT_PASS_CURRICULUM="${ENABLE_PROMPT_PASS_CURRICULUM:-False}"
export ZERO_PASS_CURRICULUM_FRACTION="${ZERO_PASS_CURRICULUM_FRACTION:-0.25}"
export PROMPT_PASS_CURRICULUM_05SORT="${PROMPT_PASS_CURRICULUM_05SORT:-True}"

seed=123
export SEED="${seed}"

export NO_RESAMPLING_PASS_RATE="${NO_RESAMPLING_PASS_RATE:-0.9}"


# Multi-node layout defaults (one learner node; vLLM uses remaining nodes)
export NUM_LEARNERS_PER_NODE="${NUM_LEARNERS_PER_NODE:-4}"
export VLLM_NUM_ENGINES="${VLLM_NUM_ENGINES:-8}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
# Enable eager if you see vLLM issues across nodes (slower but safer).
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-False}"

# Experiment name
now=$(date +%s)
hours=$(date -d @"$now" -u +%H)
minutes=$(date -d @"$now" -u +%M)
seconds=$(date -d @"$now" -u +%S)

export EXP_NAME="${EXP_NAME:-${name}_}"

bash /home/rberger/rlvr/GRPO_prompt_replay/open-instruct/scripts/train/olmo3/qwen_math_7_multinode.sh "$@"
