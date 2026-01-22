# GPU usage and monitoring for `run_grpo.slurm`

## How GPUs are allocated
- Slurm requests four H100s per node in `open-instruct/run_grpo.slurm` via `#SBATCH --gres=gpu:h100:4`. Ray and DeepSpeed then split them.
- Learners (training): `--num_learners_per_node` is a list of GPU counts per node. World size = sum of the list, and each learner uses one GPU. Example: `--num_learners_per_node 2` on a 4‑GPU node reserves two GPUs for training.
- Actors/inference (vLLM): GPUs = `vllm_num_engines * vllm_tensor_parallel_size`. Each engine is one Ray actor. In the current script `--vllm_num_engines 2 --vllm_tensor_parallel_size 1` consumes two GPUs.
- Placement: learners use a placement group with `STRICT_SPREAD` over the bundles you specify, so a two‑element list spreads learners across two nodes. vLLM engines use a separate placement group with `PACK`, so they fill whatever GPUs remain. Keep `learner GPUs + vLLM GPUs ≤ GPUs provisioned`.

## Example layouts
- Single node, 4 GPUs (current run): `--num_learners_per_node 2` (2 GPUs) + `--vllm_num_engines 2 --vllm_tensor_parallel_size 1` (2 GPUs) → all four GPUs used, clean split train vs inference.
- Single node, 4 GPUs, heavier inference: `--num_learners_per_node 1 --vllm_num_engines 3` uses 1 GPU for training, 3 for actors.
- Two nodes, 4 GPUs each, 3 GPUs for learners and 5 for actors: request `#SBATCH --nodes=2 --gres=gpu:h100:4` and pass `--num_learners_per_node 2 1` (2 learners on node0, 1 on node1). That leaves 2 free on node0 + 3 free on node1; set `--vllm_num_engines 5 --vllm_tensor_parallel_size 1` to occupy the remaining five GPUs. Ray will pack inference engines onto the free GPUs.
- Multi-node symmetric: `--num_learners_per_node 2 2` on two nodes leaves two free GPUs on each node; pair with `--vllm_num_engines 4 --vllm_tensor_parallel_size 1` (one actor per free GPU). For tensor parallel actors, multiply: `--vllm_num_engines 2 --vllm_tensor_parallel_size 2` would also consume those four free GPUs.

## How to verify GPU usage live
- On the allocation: `srun --nodes=1 --ntasks=1 watch -n1 nvidia-smi` (or `nvidia-smi pmon -s um` for per‑process utilization). Expect to see `PolicyTrainerRayProcess` (learners) and `LLMRayActor/EngineCore` (vLLM) processes occupying the intended GPUs.
- Ray view of resources from the head node shell: `ray status` and `ray list actors` show which actors are up and their resource requirements.
- Actor queue dashboard: enabled by default (`enable_queue_dashboard=True`). The stdout log prints `Dashboard server started at http://<host>:<port>`; visit or SSH‑tunnel to that URL to see queue sizes and token throughput per engine.
- Ray dashboard (heavier): export `RAY_DASHBOARD_ENABLED=1` before launch to enable the Ray web UI (port 8265 by default) if your cluster/network allows.

## Logs and metrics to watch
- Stdout log: `outputs/olmo3_final_pres/base_%j.out` shows vLLM init, placement, and evaluation/training metrics. After each step, a one-line metric summary is printed.
- WandB (with `--with_tracking` already set):
  - Throughput/utilization: `actor_tokens_per_second`, `learner_tokens_per_second_step`, `actor_mfu`, `actor_mbu`, `learner_mfu`.
  - Latency: `time/training`, `time/getting_response`, and queue dashboard stats.
  - Resource assumptions: MFU/MBU use H100 specs from `open_instruct/utils.py`, so values reflect expected vs theoretical peak.
- If MFU is low and GPUs look idle, increase batch (`num_samples_per_prompt_rollout`, `num_unique_prompts_rollout`) or inference engines, or reduce `vllm_gpu_memory_utilization` only if you need memory headroom.

## Quick checklist
- Keep `num_learners_per_node` lengths equal to `--nodes`; sum matches desired training GPUs.
- Ensure `vllm_num_engines * vllm_tensor_parallel_size` fits the remaining GPUs.
- Use `nvidia-smi` (per node) plus WandB metrics to confirm learners and actors are both busy.
