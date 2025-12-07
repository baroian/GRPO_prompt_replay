module purge

module load 2024
module load CUDA/12.6.0

export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export UV_LINK_MODE=${UV_LINK_MODE:-copy}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/scratch-shared/rberber/cache/uv}
export UV_NO_DEV=${UV_NO_DEV:-1}
export RAY_DASHBOARD_ENABLED=${RAY_DASHBOARD_ENABLED:-0}


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

