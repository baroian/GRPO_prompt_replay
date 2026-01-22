# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------
# Part of the code is adapted from https://github.com/OpenRLHF/OpenRLHF
# which has the following license:
# Copyright [yyyy] [name of copyright owner]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# isort: off
from __future__ import annotations
import contextlib
import os
from concurrent import futures

os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA
with contextlib.suppress(Exception):
    import deepspeed

from open_instruct import utils

# isort: on
import asyncio
import heapq
import json
import logging
import math
import random
import shutil
import socket
import threading
import time
from argparse import Namespace
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from queue import Empty, Full, Queue
from typing import Any, Literal

import datasets
import numpy as np
import pandas as pd
import ray
import torch
import torch.distributed as dist
import torch.utils
import torch.utils.data
import vllm
from datasets import Dataset
from huggingface_hub import HfApi
from peft import PeftModel, get_peft_model_state_dict
from ray.util import queue as ray_queue
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rich.pretty import pprint
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer, get_scheduler
from transformers.integrations import HfDeepSpeedConfig

import wandb
from open_instruct import logger_utils, vllm_utils
from open_instruct.actor_manager import ActorManager
from open_instruct.dataset_transformation import (
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    RAW_PROMPT_KEY,
    VERIFIER_SOURCE_KEY,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
from open_instruct.ground_truth_utils import (
    build_all_verifiers,
    cleanup_all_llm_judge_clients,
    soft_format_reward_func,
)
from open_instruct.model_utils import (
    Batch,
    ModelConfig,
    apply_verifiable_reward,
    disable_dropout_in_model,
    entropy_from_logits,
    get_olmo3_generation_config,
    load_ref_policy,
    log_softmax_and_gather,
    print_rich_single_line_metrics,
    print_rich_table,
    push_folder_to_hub,
)
from open_instruct.queue_types import GenerationResult, PromptRequest, RequestInfo, TokenStatistics
from open_instruct.rl_utils import PackedSequences, Timer, pack_sequences
from open_instruct.utils import (
    ArgumentParserPlus,
    BeakerRuntimeConfig,
    RayProcess,
    _z3_params_to_fetch,
    calibrate_checkpoint_state_dir,
    clean_last_n_checkpoints_deepspeed,
    combine_reward_metrics,
    download_latest_checkpoint_from_gs,
    get_beaker_whoami,
    get_eval_ds_config,
    get_optimizer_grouped_parameters,
    get_train_ds_config,
    get_wandb_tags,
    is_beaker_job,
    launch_ai2_evals_on_weka,
    maybe_get_beaker_config,
    maybe_update_beaker_description,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
    ray_get_with_progress,
    repeat_each,
    sync_gs_bucket,
)

logger = logger_utils.setup_logger(__name__)

INVALID_LOGPROB = 1.0


class ShutdownSentinel:
    """Sentinel value to signal thread shutdown via queue."""


@dataclass
class Args:
    # Dataset
    dataset_mixer_list: list[str] = field(default_factory=lambda: ["ai2-adapt-dev/rlvr_gsm8k_zs", "1.0"])
    """A list of datasets (local or HF) to sample from."""
    dataset_mixer_eval_list: list[str] = field(default_factory=lambda: ["ai2-adapt-dev/rlvr_gsm8k_zs", "1.0"])
    """A list of datasets (local or HF) to sample from for evaluation."""
    dataset_mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])
    """The dataset splits to use for training"""
    dataset_mixer_eval_list_splits: list[str] = field(default_factory=lambda: ["test"])
    """The dataset splits to use for evaluation"""
    local_eval_subset_sample_count: int = 256
    """If > 0, derive evaluation datasets by sampling this many prompts from the training mixture."""
    local_eval_timeout: float = 0.01
    """Maximum seconds to wait for local eval generations during training steps."""
    dataset_mixer_benchmark_list: list[str] = field(default_factory=list)
    """A list of datasets (local or HF) to sample from for benchmark evaluation (separate from local eval subset)."""
    dataset_mixer_benchmark_list_splits: list[str] = field(default_factory=list)
    """The dataset splits to use for benchmark evaluation."""
    benchmark_eval_every: int = -1
    """Run benchmark evaluation after this many training steps. Set to -1 to use local_eval_every. Set to 0 to disable."""
    dataset_config_benchmark_hash: str | None = None
    """The hash of the dataset configuration for benchmark evaluation."""
    dataset_transform_fn_benchmark: list[str] | None = None
    """Transform functions for benchmark datasets. If None, uses dataset_transform_fn.
    Useful when benchmark datasets have different column formats than training data.
    Common values: ["convert_math500_format", "rlvr_tokenize_v1", "rlvr_max_length_filter_v1"]
    for MATH-500 style datasets, or ["convert_minervamath_format", "rlvr_tokenize_v1", "rlvr_max_length_filter_v1"]
    for minervamath style datasets."""
    dataset_transform_fn: list[str] = field(default_factory=lambda: ["rlvr_tokenize_v1", "rlvr_max_length_filter_v1"])
    """The list of transform functions to apply to the dataset."""
    dataset_cache_mode: Literal["hf", "local"] = "local"
    """The mode to use for caching the dataset."""
    dataset_local_cache_dir: str = "local_dataset_cache"
    """The directory to save the local dataset cache to."""
    dataset_config_hash: str | None = None
    """The hash of the dataset configuration."""
    dataset_config_eval_hash: str | None = None
    """The hash of the dataset configuration for evaluation."""
    dataset_skip_cache: bool = False
    """Whether to skip the cache."""
    shuffle_eval_dataset: bool = False
    """Whether to shuffle the evaluation dataset."""
    max_prompt_token_length: int = 256
    """The maximum prompt token length to use for the dataset"""
    system_prompt_override_file: str | None = None
    """Path to a text file containing a system prompt to override the dataset's system prompts"""
    enable_prompt_pass_curriculum: bool = False
    """Whether to reorder prompts each epoch using prompt pass rates."""
    zero_pass_curriculum_fraction: float = 0.25
    """Fraction of zero-pass prompts from the previous epoch to schedule in the next epoch."""
    prompt_pass_curriculum_05sort: bool = False
    """If True, prioritize prompts closest to 50% pass rate (ties favor lower pass rates)."""
    enable_prompt_replay: bool = False
    """Whether to reuse high pass-rate prompts each training step."""
    prompt_replay_fraction: float = 0.5
    """Maximum fraction of each batch that can be filled with replayed prompts."""
    prompt_replay_cooldown_steps: int = 5
    """Number of training steps a prompt must wait before it becomes eligible for replay again."""
    prompt_replay_max_reuse_time: int = 20
    """Maximum number of times a prompt can be replayed before it is retired. Set to 0 or a negative value to disable the limit."""
    prompt_replay_min_pass_rate: float = 0.24
    """Minimum pass rate (inclusive) required for a prompt to enter replay. Set to a negative value to disable the lower bound."""
    prompt_replay_max_pass_rate: float = 0.7
    """Maximum pass rate (inclusive) allowed for a prompt to enter replay. Set to a negative value to disable the upper bound."""

    # Experiment
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    seed: int = 1
    """Seed of the experiment"""
    run_name: str | None = None
    """RUNTIME VALUE: A unique name of this run"""

    # Optimizer
    learning_rate: float = 2e-5
    """The initial learning rate for AdamW optimizer."""
    lr_scheduler_type: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ] = "linear"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""
    warmup_ratio: float = 0.0
    """Ratio of warmup steps to total steps (takes precedence over `warm_up_steps`)"""
    weight_decay: float = 0.0
    """Weight decay for AdamW if we apply some."""
    set_weight_decay_on_bias_and_norm: bool = True
    """Whether to set weight decay on bias and norm layers"""
    fused_optimizer: bool = False
    """Whether to use fused optimizer"""

    # Batch sizes
    per_device_train_batch_size: int = 1
    """The forward batch size per device (local_micro_batch_size)"""
    total_episodes: int = 100000
    """The total number of episodes in the dataset"""
    world_size: int | None = None
    """RUNTIME VALUE: The number of processes (GPUs) to use"""
    num_training_steps: int | None = None
    """RUNTIME VALUE: The number of training_steps to train"""
    local_eval_every: int = 100
    """Run evaluation after this many training steps. This controls in-loop evals, which reuse the generation/reward verifier setup. Set to -1 to disable."""
    save_freq: int = 200
    """How many train steps to save the model"""
    allow_world_padding: bool = False
    """Whether to allow world padding. This is useful for model sweeps, but wastes compute."""
    backend_timeout: int = 120
    """Timeout for inference/training backends in minutes. Default is 2 hours (120 min)."""

    # Generation
    response_length: int = 256
    """the length of the response"""
    temperature: float = 0.7
    """the sampling temperature"""
    num_unique_prompts_rollout: int = 16
    """The number of unique prompts during rollout"""
    num_samples_per_prompt_rollout: int = 4
    """the number of samples to generate per prompt during rollout, useful for easy-star"""
    stop_strings: list[str] | None = None
    """List of strings that stop the generation when they are generated.
    The returned output will not contain the stop strings."""
    # Algorithm
    async_steps: int = 1
    """Number of steps ahead to generate responses. Fully synchronous training is not supported, so async_steps must be greater than 0. The trainer learns from a policy up to async_steps old like Cleanba (https://arxiv.org/abs/2310.00036)"""
    num_epochs: int = 1
    """the number of epochs to train"""
    num_mini_batches: int = 1
    """Number of minibatches to split a batch into"""
    beta: float = 0.05
    """the beta value of the RLHF objective (KL coefficient)"""
    clip_lower: float = 0.2
    """the lower clip range"""
    clip_higher: float = 0.2
    """the higher clip range. Sometimes we want this to be higher, see DAPO (https://arxiv.org/abs/2503.14476)"""
    truncated_importance_sampling_ratio_cap: float = 0.0
    """The maximum cap for truncated importance sampling ratio (0 means disabled)"""
    inflight_updates: bool = False
    """Enable immediate stopping of request processing when should_stop is set, allowing for quick pausing and resumption"""
    kl_estimator: Literal["kl1", "kl2", "kl3", "kl4"] = "kl3"
    """the KL estimator to use"""
    pack_length: int = 512
    """the length of the pack (you should prob set to the max length of the model)"""
    masked_mean_axis: int | None = None
    """the axis to compute the mean of the masked values"""
    masked_mean_denominator: float | None = None
    """Optional constant denominator for masked_mean; if set, divides by this instead of mask.sum"""
    alpha: float = 0.6
    """The alpha value for doing polyak updates (ref_param = alpha * param + (1 - alpha) * ref_param)
    reference: [TR-DPO](https://huggingface.co/papers/2404.09656), but it's actually pretty commonly
    used. E.g., [TD3](https://arxiv.org/abs/1802.09477) uses https://github.com/vwxyzjn/cleanrl/blob/dcc289fc6f0bda492fa7360a155262cf826b12a5/cleanrl/td3_continuous_action.py#L269
    """
    ref_policy_update_freq: int | None = None
    """How many training steps to take before updating the reference policy."""
    load_ref_policy: bool = True
    """Whether to load and use a reference policy for KL penalty calculation."""
    advantage_normalization_type: Literal["standard", "centered"] = "standard"
    """The type of advantage normalization to use. Standard normalization is the default: it subtracts the mean and
    divides by the standard deviation. Centered normalization is the same but subtracts the mean only (e.g., used in
    DR.GRPO https://arxiv.org/pdf/2503.20783)."""
    mask_truncated_completions: bool = False
    """Whether to mask out truncated completions. Also called overlong filtering, from DAPO (https://arxiv.org/abs/2503.14476)."""

    active_sampling: bool = False
    """Whether to continue sampling responses until you get a full batch."""
    filter_zero_std_samples: bool = True
    """Whether to filter out prompts with zero reward std (all samples have the same score)."""
    no_resampling_pass_rate: float | None = None
    """If the response to a prompt is solved at a rate higher than this, do not resample this prompt again"""

    record_entropy: bool = False
    """whether to record the entropy of the policy during training. Uses extra memory."""
    use_vllm_logprobs: bool = False
    """whether to use vLLM's logprobs for training instead of calculating them via forward pass"""

    # Reward
    # -- r1 style format reward
    apply_r1_style_format_reward: bool = False
    """whether to add the R1 style format reward"""
    r1_style_format_reward: float = 1.0
    """the reward value for R1 style format reward"""
    additive_format_reward: bool = False
    """whether to add the format reward to the final reward"""

    # -- verifiable reward
    apply_verifiable_reward: bool = True
    """whether to apply verifiable reward"""
    verification_reward: float = 10.0
    """the reward value for verifiable responses"""
    remap_verifier: str = None
    """Remap verifier like string_f1=general-quality_ref. Currently can only remap once."""

    # -- llm verifiers
    llm_judge_model: str = "azure/gpt-4o-mini-standard"
    """the model to use for the llm judge"""
    llm_judge_max_tokens: int = 2048
    """the max tokens to use for the llm judge"""
    llm_judge_max_context_length: int = 8192
    """the max context length to use for the llm judge"""
    llm_judge_temperature: float = 1.0
    """the temperature to use for the llm judge"""
    llm_judge_timeout: int = 60
    """the timeout to use for the llm judge"""

    # -- code verifier
    code_api_url: str = os.environ.get("CODE_API_URL", "http://localhost:1234") + "/test_program"
    """the api url to use for the code verifier"""
    code_max_execution_time: float = 1.0
    """the max execution time to use for the code verifier"""
    code_pass_rate_reward_threshold: float = 0.0
    """the pass rate reward threshold for the code verifier. If pass rate is less than this threshold, reward is 0.0, otherwise reward is pass rate"""
    code_apply_perf_penalty: bool = False
    """whether to apply a performance penalty to the code verifier"""

    # -- max length verifier
    max_length_verifier_max_length: int = 32768
    """the max length to use for the max length verifier"""

    # -- non stop penalty
    non_stop_penalty: bool = False
    """whether to penalize responses which did not finish generation"""
    non_stop_penalty_value: float = 0.0
    """the reward value for responses which did not finish generation"""

    # Ray
    single_gpu_mode: bool = False
    """whether to collocate vLLM and actor on the same node (mostly for debugging purposes)"""
    num_learners_per_node: list[int] = field(default_factory=lambda: [1])
    """number of GPU deepspeed learners per node (e.g., --num_learners_per_node 2 4 means 2 learner processes
    on the first node and 4 learner processes on the second node; each process will have 1 GPU)"""
    vllm_num_engines: int = 1
    """number of vLLM Engines, set to 0 to disable vLLM"""
    vllm_tensor_parallel_size: int = 1
    """tensor parallel size of vLLM Engine for multi-GPU inference"""
    vllm_enforce_eager: bool = False
    """whether to enforce eager mode for vLLM -- slow inference but needed for multi-node"""
    vllm_sync_backend: str = "nccl"
    """DeepSpeed -> vLLM weight sync backend"""
    vllm_gpu_memory_utilization: float = 0.9
    """vLLM GPU memory utilization"""
    vllm_enable_prefix_caching: bool = False
    """whether to enable prefix caching"""
    vllm_top_p: float = 1.0
    """vLLM top p for nucleus sampling"""
    deepspeed_stage: int = 0
    """the deepspeed stage"""
    deepspeed_zpg: int = 8
    """the deepspeed zpg value. Higher values are more memory efficient but slower. Set to 1 to disable zpg, which uses less memory but is significantly slower. Ideally is set to the number of GPUs per node (usually 8, default)."""
    deepspeed_offload_param: bool = False
    """whether to offload parameters to CPU (reduces GPU memory usage)"""
    deepspeed_offload_optimizer: bool = False
    """whether to offload optimizer states to CPU (reduces GPU memory usage)"""
    gather_whole_model: bool = True
    """whether to gather the whole model to boardcast (not doable for 70B but can be faster for 8B)"""
    enable_queue_dashboard: bool = True
    """whether to enable the ActorManager queue monitoring dashboard"""
    queue_dashboard_port: int | None = None
    """optional port for the dashboard server (if None, finds a free port automatically)"""

    # Experiment tracking
    verbose: bool = False
    """If toggled, debug output will be shown"""
    update_progress_every: int = 10
    """How often to update the progress bar (in steps)."""
    with_tracking: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    prompt_pass_table_dir: str | None = None
    """If set, log per-prompt pass rates each training step and store them under this directory"""
    wandb_project_name: str = "open_instruct_internal"
    """The wandb's project name"""
    wandb_entity: str | None = None
    """The entity (team) of wandb's project"""
    push_to_hub: bool = True
    """Whether to upload the saved model to huggingface"""
    hf_entity: str | None = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: str | None = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: str | None = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: str | None = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    output_dir: str = "output"
    """Where to save the model"""
    save_traces: bool = False
    """Whether to save learning data traces"""
    cache_dataset_only: bool = False
    """Immediately exit after caching the dataset"""
    keep_last_n_checkpoints: int = 3
    """How many checkpoints to keep in the output directory. -1 for all."""
    checkpoint_state_freq: int = -1
    """How often to save the model checkpoint, optimizer states, and lr scheduler states (in steps)"""
    checkpoint_state_dir: str | None = None
    """Where to save the model checkpoint (if applicable)"""
    gs_checkpoint_state_dir: str | None = None
    """The actual `checkpoint_state_dir` to use (handling the case where gs_bucket_path is provided)"""

    # Ai2 specific settings
    try_launch_beaker_eval_jobs_on_weka: bool = False
    """Whether to launch beaker evaluation jobs after training on weka"""
    try_auto_save_to_beaker: bool = True
    """Whether to try to save the model to Beaker dataset `/output` after training"""
    gs_bucket_path: str | None = None
    """The path to the gs bucket to save the model to"""
    oe_eval_tasks: list[str] | None = None
    """The beaker evaluation tasks to launch"""
    oe_eval_max_length: int = 4096
    """the max generation length for evaluation for oe-eval"""
    oe_eval_beaker_image: str | None = None
    """the docker image for evaluation for oe-eval"""
    oe_eval_gpu_multiplier: int | None = None
    """multiply the gpus used for each oe-eval task"""
    eval_priority: Literal["low", "normal", "high", "urgent"] = "normal"
    """the priority of auto-launched evaluation jobs"""
    eval_workspace: str = "ai2/tulu-3-results"
    """the workspace to launch evaluation jobs on"""
    send_slack_alerts: bool = False
    """Whether to send Slack alerts on training failures"""

    # Evaluation behavior
    eval_on_step_0: bool = False
    """Whether to run local evaluation at training step 0. Defaults to False."""

    # Tool settings
    tools: list[str] | None = None
    """If set, use the tool mapped to the string. Currently only supports `search` and `code`"""
    max_tool_calls: list[int] = field(default_factory=lambda: [5])
    """Maximum number of tool calls allowed. If a list is provided, it must have length 1 (applies to all tools) or same length as tools (per-tool limit)."""
    mask_tool_use: bool = True
    """Whether to mask the tool output. By default on."""
    only_reward_good_outputs: bool = False
    """Whether to only reward good outputs. By default off. Useful to force the model to use the tool(s)."""

    # rl-rag specific settngs
    number_documents_to_search: int = 3
    """The maximum number of documents to retrieve for each query."""
    search_api_endpoint: str | None = None
    """The API endpoint for the search engine."""

    # code-tool specific settings
    code_tool_api_endpoint: str | None = None

    def __post_init__(self):
        if os.environ.get("VLLM_USE_V1") == "0":
            logger.warning("When using the v0 version of vLLM, caching is broken and will never be invalidated.")
            if self.vllm_enable_prefix_caching:
                raise ValueError("Prefix caching is currently not supported for v0.")
        if self.use_vllm_logprobs and self.truncated_importance_sampling_ratio_cap > 0.0:
            raise ValueError(
                "Cannot use both `use_vllm_logprobs` and `truncated_importance_sampling_ratio_cap`. "
                "use_vllm_logprobs sets old_logprobs to vLLM logprobs, making importance sampling pointless."
            )
        if self.masked_mean_denominator is not None:
            assert self.masked_mean_denominator > 0, (
                f"masked_mean_denominator (={self.masked_mean_denominator}) must be greater than 0!"
            )
        assert self.num_samples_per_prompt_rollout > 0, "Number of samples per prompt must be greater than 0!"
        if self.num_samples_per_prompt_rollout == 1:
            logger.warning("num_samples_per_prompt_rollout is 1. This reduces GRPO to REINFORCE.")
        assert self.apply_verifiable_reward or self.apply_r1_style_format_reward or self.non_stop_penalty, (
            "At least one reward must be applied!"
        )
        # Ensure we have enough prompts for all VLLM engines
        if self.num_unique_prompts_rollout < self.vllm_num_engines:
            logger.warning(
                f"With num_unique_prompts_rollout={self.num_unique_prompts_rollout} < "
                f"vllm_num_engines={self.vllm_num_engines}, vllm will be generating data for multiple "
                "batches simultaneously. This is fine but might be unexpected behaviour."
            )
        # Initialize stop_strings if None
        if self.stop_strings is None:
            self.stop_strings = []
        assert self.pack_length >= self.max_prompt_token_length + self.response_length, (
            "The `pack_length` needs to be greater than the sum of `max_prompt_token_length` and `response_length`!"
        )
        if self.checkpoint_state_freq > 0 and self.checkpoint_state_dir is None:
            raise ValueError("`checkpoint_state_dir` must be provided if `checkpoint_state_freq` is greater than 0!")
        if self.checkpoint_state_dir is not None and self.checkpoint_state_freq == -1:
            raise ValueError("`checkpoint_state_freq` must be greater than 0 if `checkpoint_state_dir` is provided!")

        if self.gs_checkpoint_state_dir is not None and not self.gs_checkpoint_state_dir.startswith("gs://"):
            raise ValueError(f"`gs_checkpoint_state_dir` must start with 'gs://', got: {self.gs_checkpoint_state_dir}")
        if self.gs_bucket_path is not None and not self.gs_bucket_path.startswith("gs://"):
            raise ValueError(f"`gs_bucket_path` must start with 'gs://', got: {self.gs_bucket_path}")

        if self.gs_bucket_path is not None and self.gs_checkpoint_state_dir is None:
            if self.checkpoint_state_dir is None:
                raise ValueError("`checkpoint_state_dir` must be provided when using `gs_bucket_path`!")
            checkpoint_dir_name = self.checkpoint_state_dir.rstrip("/")
            beaker_users = get_beaker_whoami()
            if beaker_users is not None:
                self.gs_checkpoint_state_dir = f"{self.gs_bucket_path}/{beaker_users}/{checkpoint_dir_name}"
            else:
                self.gs_checkpoint_state_dir = f"{self.gs_bucket_path}/{checkpoint_dir_name}"

        if self.checkpoint_state_dir is not None:
            if self.gs_checkpoint_state_dir is not None:
                download_latest_checkpoint_from_gs(self.gs_checkpoint_state_dir, self.checkpoint_state_dir)
            calibrate_checkpoint_state_dir(self.checkpoint_state_dir)
        if self.tools is not None and len(self.tools) > 0:
            for tool in self.tools:
                if tool not in ["search", "code"]:
                    raise ValueError(f"Tool {tool} is not supported. Supported tools are: search, code")
            assert len(self.tools) == len(set(self.tools)), "Duplicate tools are not allowed"
            if self.use_vllm_logprobs or self.truncated_importance_sampling_ratio_cap > 0.0:
                assert self.mask_tool_use, (
                    "Must mask tool use when using vLLM logprobs or truncated importance sampling."
                )
        if not self.load_ref_policy and self.beta != 0.0:
            raise ValueError(
                "When load_ref_policy=False, beta must be 0.0. "
                f"Got beta={self.beta}. Set --beta 0.0 or --load_ref_policy to use KL penalty."
            )

        # Figure out max possible RLVR score
        self.max_possible_score = 0
        if self.apply_verifiable_reward:
            self.max_possible_score += self.verification_reward
        if self.apply_r1_style_format_reward and self.additive_format_reward:
            self.max_possible_score += self.r1_style_format_reward

        if self.active_sampling:
            assert self.async_steps > 1, (
                "With active_sampling, you should set async_steps > 1 to account for filtering of the first batch. "
                "Otherwise, your generator only generates only one batch worth of prompts and a single filtered "
                "prompt will cause the trainer to stall waiting for more data  . "
            )
            assert self.filter_zero_std_samples, (
                "filter_zero_std_samples must be True when active_sampling is True. "
                "Active sampling requires filtering to work correctly."
            )
        if not (0.0 <= self.zero_pass_curriculum_fraction <= 1.0):
            raise ValueError(
                f"`zero_pass_curriculum_fraction` must be between 0 and 1 inclusive, "
                f"got {self.zero_pass_curriculum_fraction}"
            )
        if self.num_samples_per_prompt_rollout == 1 and self.filter_zero_std_samples:
            raise ValueError(
                "`filter_zero_std_samples` cannot be True when `num_samples_per_prompt_rollout` is 1, "
                "as the reward standard deviation will always be 0, causing all samples to be filtered."
            )
        if self.async_steps < 1:
            raise ValueError("`async_steps` must be greater than 0. Fully synchronous training is not supported.")


def masked_mean(
    values: torch.Tensor, mask: torch.Tensor, axis: int | None = None, denominator: float | None = None
) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    numerator = (values * mask).sum(axis=axis)
    denom = mask.sum(axis=axis) if denominator is None else denominator
    return (numerator / denom).mean()


def collate_fn(tensors_list: list[torch.Tensor], pad_token_id: int, pin_memory: bool = True) -> torch.Tensor:
    padded_tensor = torch.nn.utils.rnn.pad_sequence(tensors_list, batch_first=True, padding_value=pad_token_id)
    if pin_memory:
        padded_tensor = padded_tensor.pin_memory()
    return padded_tensor


@Timer("🔄 [Data Preparation Thread] Prepare collated data for each worker")
def prepare_collated_data_for_workers(
    packed_sequences: PackedSequences,
    world_size: int,
    per_device_train_batch_size: int,
    pad_token_id: int,
    pin_memory: bool = True,
) -> list[dict[str, list[torch.Tensor]]]:
    """Distributes and collates packed sequences for distributed training.

    Splits packed sequences across workers, randomly shuffles each worker's data,
    and collates into micro-batches for training.

    Args:
        packed_sequences: Packed training sequences containing query responses,
            tool masks, attention masks, position IDs, advantages, response masks,
            and vllm logprobs.
        world_size: Number of distributed workers.
        per_device_train_batch_size: Batch size for each device's micro-batch.
        pad_token_id: Token ID used for padding sequences.
        pin_memory: Whether to pin memory for faster data transfer to GPU.

    Returns:
        List of dictionaries, one per worker, each containing collated tensors
        for query_responses, tool_masks, attention_masks, position_ids,
        advantages, response_masks, and vllm_logprobs.
    """
    B = len(packed_sequences.query_responses) // world_size  # essentially doing `drop_last=True`, which is fine.
    collated_data = []
    for i in range(world_size):
        per_device_packed_query_responses = packed_sequences.query_responses[B * i : B * (i + 1)]
        per_device_packed_tool_masks = packed_sequences.tool_masks[B * i : B * (i + 1)]
        per_device_packed_attention_masks = packed_sequences.attention_masks[B * i : B * (i + 1)]
        per_device_packed_position_ids = packed_sequences.position_ids[B * i : B * (i + 1)]
        per_device_packed_advantages = packed_sequences.advantages[B * i : B * (i + 1)]
        per_device_packed_response_masks = packed_sequences.response_masks[B * i : B * (i + 1)]
        per_device_packed_vllm_logprobs = packed_sequences.vllm_logprobs[B * i : B * (i + 1)]

        # Shuffle the batch and collate the data
        b_inds = np.random.permutation(len(per_device_packed_query_responses))
        collated_query_responses = []
        collated_tool_masks = []
        collated_attention_masks = []
        collated_position_ids = []
        collated_response_masks = []
        collated_advantages = []
        collated_vllm_logprobs = []
        for j in range(0, len(per_device_packed_query_responses), per_device_train_batch_size):
            micro_range = b_inds[j : j + per_device_train_batch_size]
            collated_query_responses.append(
                collate_fn([per_device_packed_query_responses[idx] for idx in micro_range], pad_token_id, pin_memory)
            )
            collated_tool_masks.append(
                collate_fn([per_device_packed_tool_masks[idx] for idx in micro_range], 0, pin_memory)
            )
            collated_attention_masks.append(
                collate_fn([per_device_packed_attention_masks[idx] for idx in micro_range], 0, pin_memory)
            )
            collated_position_ids.append(
                collate_fn([per_device_packed_position_ids[idx] for idx in micro_range], 0, pin_memory)
            )
            collated_response_masks.append(
                collate_fn([per_device_packed_response_masks[idx] for idx in micro_range], 0, pin_memory)
            )
            collated_advantages.append(
                collate_fn([per_device_packed_advantages[idx] for idx in micro_range], 0, pin_memory)
            )
            collated_vllm_logprobs.append(
                collate_fn([per_device_packed_vllm_logprobs[idx] for idx in micro_range], 0, pin_memory)
            )
        collated_data.append(
            {
                "collated_query_responses": collated_query_responses,
                "collated_tool_masks": collated_tool_masks,
                "collated_attention_masks": collated_attention_masks,
                "collated_position_ids": collated_position_ids,
                "collated_advantages": collated_advantages,
                "collated_response_masks": collated_response_masks,
                "collated_vllm_logprobs": collated_vllm_logprobs,
            }
        )
    return collated_data


def to_device_inplace(tensors_list: list[torch.Tensor], device: torch.device):
    for i in range(len(tensors_list)):
        tensors_list[i] = tensors_list[i].to(device, non_blocking=True)


class ShufflingIterator:
    def __init__(
        self,
        data: np.ndarray,
        batch_size: int,
        seed: int | None = None,
        enable_prompt_pass_curriculum: bool = False,
        zero_pass_curriculum_fraction: float = 0.25,
        prompt_pass_curriculum_05sort: bool = False,
        prompts_per_step: int = 1,
        enable_prompt_replay: bool = False,
        prompt_replay_fraction: float = 0.5,
        prompt_replay_cooldown_steps: int = 5,
        prompt_replay_max_reuse_time: int = 5,
        prompt_replay_min_pass_rate: float | None = 0.24,
        prompt_replay_max_pass_rate: float | None = 0.7,
    ):
        self.base_data = data.copy()
        self.batch_size = batch_size
        self.index = 0
        self.epoch_number = 0
        self.rng = np.random.default_rng(seed)
        self.exclude_list: list[int] = []
        self.curriculum_enabled = enable_prompt_pass_curriculum
        self.zero_pass_fraction = zero_pass_curriculum_fraction
        self.curriculum_center_sort = bool(prompt_pass_curriculum_05sort)
        self._permanent_exclusions: set[int] = set()
        self.prompt_pass_rates: dict[int, float] = {}
        self.prompt_last_epoch: dict[int, int] = {}
        self.prompt_last_step: dict[int, int] = {}
        self._zero_pass_heap: list[tuple[int, int, int]] = []
        self._zero_pass_lookup: dict[int, tuple[int, int]] = {}
        self._zero_pass_counter = 0
        self._zero_pass_epochs: defaultdict[int, set[int]] = defaultdict(set)
        self.data = self.base_data.copy()
        self.prompts_per_step = max(1, prompts_per_step)
        self.replay_enabled = enable_prompt_replay
        self.prompt_stats_enabled = self.curriculum_enabled or self.replay_enabled
        self.replay_fraction = min(max(prompt_replay_fraction, 0.0), 0.5)
        self.replay_cooldown_steps = max(0, int(prompt_replay_cooldown_steps))
        if prompt_replay_max_reuse_time is None:
            self.replay_max_reuse_limit = None
        else:
            max_reuse_steps = int(prompt_replay_max_reuse_time)
            self.replay_max_reuse_limit = max_reuse_steps if max_reuse_steps > 0 else None
        self.prompt_reuse_counts: dict[int, int] = {}
        self._active_replay_indices: set[int] = set()
        self._step_states: dict[int, StepReplayState] = {}
        self.replay_min_pass_rate = self._normalize_rate_bound(prompt_replay_min_pass_rate)
        self.replay_max_pass_rate = self._normalize_rate_bound(prompt_replay_max_pass_rate)
        if (
            self.replay_min_pass_rate is not None
            and self.replay_max_pass_rate is not None
            and self.replay_min_pass_rate > self.replay_max_pass_rate
        ):
            self.replay_min_pass_rate, self.replay_max_pass_rate = (
                self.replay_max_pass_rate,
                self.replay_min_pass_rate,
            )
        self._replay_heap: list[tuple[float, float, int, int]] = []
        self._progress_reorder_enabled = self.replay_enabled
        self._target_progress_unit = 0
        self._last_reordered_progress_unit = 0
        self._pending_epoch_completions: deque[int] = deque()
        self._prepare_next_epoch(completed_epoch=-1)

    def __iter__(self) -> Iterator[list[int]]:
        return self

    def __next__(self) -> list[int] | int:
        """Return a list of next indices or a single index if batch size is 1"""
        if self.batch_size == 1:
            return self._next_index()
        batch = [self._next_index() for _ in range(self.batch_size)]
        return batch

    def _raw_next_index(self) -> int:
        if self.effective_size == 0:
            raise RuntimeError("ShufflingIterator has no data available for iteration")

        if self.index >= self.effective_size:
            completed_epoch = self.epoch_number
            self.index = 0
            self._update_effective_size()
            self.epoch_number += 1
            self._pending_epoch_completions.append(completed_epoch)
            self._maybe_prepare_epochs()

        idx = int(self.data[self.index])
        self.index += 1
        return idx

    def _next_index(self, exclude: set[int] | None = None) -> int:
        while True:
            self._maybe_prepare_epochs()
            idx = self._raw_next_index()
            if not exclude or idx not in exclude:
                return idx

    def update_progress_fraction(self, progress_fraction: float | None) -> bool:
        """Record the latest cumulative progress fraction and trigger future resorting."""
        if not self._progress_reorder_enabled or progress_fraction is None:
            return False
        target_unit = int(progress_fraction)
        if target_unit < 1 or target_unit <= self._target_progress_unit:
            return False
        self._target_progress_unit = target_unit
        return True

    def _maybe_prepare_epochs(self) -> None:
        if not self._pending_epoch_completions:
            return

        if not self._progress_reorder_enabled:
            while self._pending_epoch_completions:
                epoch = self._pending_epoch_completions.popleft()
                self._prepare_next_epoch(epoch)
            return

        while self._pending_epoch_completions and self._last_reordered_progress_unit < self._target_progress_unit:
            epoch = self._pending_epoch_completions.popleft()
            self._prepare_next_epoch(epoch)
            self._last_reordered_progress_unit += 1

    def next_for_step(self, target_step: int | None) -> tuple[int, PromptReplayMetadata]:
        """Return the next dataset index along with replay metadata."""
        if target_step is None or not self.replay_enabled or self.prompts_per_step <= 0 or self.replay_fraction <= 0.0:
            idx = self._next_index()
            return idx, PromptReplayMetadata(scheduled_step=target_step)

        state = self._get_step_state(target_step)
        candidate_metadata = None
        if state.replay_budget > 0 and state.issued_replay < state.replay_budget:
            candidate_metadata = self._pop_next_replay_candidate(target_step, state.reserved_indices)

        if candidate_metadata is not None:
            idx, scheduled_step, cooldown_ready_step = candidate_metadata
            state.issued_total += 1
            state.issued_replay += 1
            state.reserved_indices.add(idx)
            self.prompt_reuse_counts[idx] = self.prompt_reuse_counts.get(idx, 0) + 1
            self._active_replay_indices.add(idx)
            self._finalize_step_state_if_needed(target_step, state)
            return (
                idx,
                PromptReplayMetadata(
                    was_reused=True,
                    reuse_count=self.prompt_reuse_counts[idx],
                    scheduled_step=scheduled_step,
                    cooldown_ready_step=cooldown_ready_step,
                ),
            )

        idx = self._next_index(exclude=state.reserved_indices)
        state.issued_total += 1
        state.issued_new += 1
        self._finalize_step_state_if_needed(target_step, state)
        return idx, PromptReplayMetadata(was_reused=False, scheduled_step=target_step)

    def _get_step_state(self, target_step: int) -> StepReplayState:
        state = self._step_states.get(target_step)
        if state is None:
            state = StepReplayState(replay_budget=self._max_replay_budget())
            self._step_states[target_step] = state
        return state

    def _max_replay_budget(self) -> int:
        if not self.replay_enabled or self.prompts_per_step <= 1:
            return 0
        raw_budget = int(math.floor(self.prompts_per_step * self.replay_fraction))
        half_cap = self.prompts_per_step // 2
        budget = min(raw_budget, half_cap)
        return max(0, budget)

    def _finalize_step_state_if_needed(self, target_step: int, state: StepReplayState) -> None:
        if state.issued_total >= self.prompts_per_step:
            self._step_states.pop(target_step, None)

    def _pop_next_replay_candidate(self, target_step: int, reserved_indices: set[int]) -> tuple[int, int, int] | None:
        temp: list[tuple[float, float, int, int]] = []
        candidate: tuple[int, int, int] | None = None
        while self._replay_heap:
            distance, prioritized_rate, reuse_snapshot, dataset_index = heapq.heappop(self._replay_heap)
            rate = self.prompt_pass_rates.get(dataset_index)
            if rate is None or not self._is_rate_within_replay_window(rate):
                continue
            if dataset_index in self._permanent_exclusions or dataset_index in reserved_indices:
                continue
            if dataset_index in self._active_replay_indices:
                temp.append((distance, prioritized_rate, reuse_snapshot, dataset_index))
                continue
            current_reuse = self.prompt_reuse_counts.get(dataset_index, 0)
            if current_reuse != reuse_snapshot:
                refreshed_entry = self._make_replay_heap_entry(dataset_index, rate)
                if refreshed_entry is not None:
                    heapq.heappush(self._replay_heap, refreshed_entry)
                continue
            if not self._is_prompt_cooled_down(dataset_index, target_step):
                temp.append((distance, prioritized_rate, reuse_snapshot, dataset_index))
                continue
            if not self._is_prompt_within_reuse_window(dataset_index, target_step):
                # Drop prompts that have exhausted their reuse budget.
                continue
            scheduled_step = target_step
            cooldown_ready_step = (
                target_step + self.replay_cooldown_steps if self.replay_cooldown_steps > 0 else target_step
            )
            candidate = (dataset_index, scheduled_step, cooldown_ready_step)
            break

        for item in temp:
            heapq.heappush(self._replay_heap, item)
        return candidate

    def _is_prompt_cooled_down(self, dataset_index: int, target_step: int) -> bool:
        if self.replay_cooldown_steps <= 0:
            return True
        last_step = self.prompt_last_step.get(dataset_index)
        if last_step is None:
            return True
        return (target_step - last_step) >= self.replay_cooldown_steps

    def _is_prompt_within_reuse_window(self, dataset_index: int, target_step: int) -> bool:
        if self.replay_max_reuse_limit is None:
            return True
        current_reuse = self.prompt_reuse_counts.get(dataset_index, 0)
        return current_reuse < self.replay_max_reuse_limit

    def _push_replay_candidate(self, dataset_index: int) -> None:
        if not self.replay_enabled:
            return
        rate = self.prompt_pass_rates.get(dataset_index)
        if rate is None or not self._is_rate_within_replay_window(rate):
            return
        if self.replay_max_reuse_limit is not None:
            current_reuse = self.prompt_reuse_counts.get(dataset_index, 0)
            if current_reuse >= self.replay_max_reuse_limit:
                return
        entry = self._make_replay_heap_entry(dataset_index, rate)
        if entry is not None:
            heapq.heappush(self._replay_heap, entry)

    @staticmethod
    def _normalize_rate_bound(value: float | None) -> float | None:
        if value is None:
            return None
        value = float(value)
        if value < 0.0:
            return None
        return float(min(max(value, 0.0), 1.0))

    def _is_rate_within_replay_window(self, rate: float) -> bool:
        if rate <= 0.0:
            return False
        if self.replay_min_pass_rate is not None and rate < self.replay_min_pass_rate:
            return False
        if self.replay_max_pass_rate is not None and rate > self.replay_max_pass_rate:
            return False
        return True

    def _make_replay_heap_entry(
        self, dataset_index: int, rate: float | None = None
    ) -> tuple[float, float, int, int] | None:
        if rate is None:
            rate = self.prompt_pass_rates.get(dataset_index)
        if rate is None or not self._is_rate_within_replay_window(rate):
            return None
        distance = abs(rate - 0.5)
        reuse_snapshot = self.prompt_reuse_counts.get(dataset_index, 0)
        return (distance, rate, reuse_snapshot, dataset_index)

    def _rebuild_replay_heap(self) -> None:
        if not self.replay_enabled:
            self._replay_heap = []
            return
        self._replay_heap = []
        for dataset_index in self.prompt_pass_rates.keys():
            self._push_replay_candidate(dataset_index)

    def get_state(self) -> dict[str, Any]:
        """Get the current state of the iterator for checkpointing."""
        state = {
            "index": self.index,
            "epoch_number": self.epoch_number,
            "data": self.data.copy(),
            "rng_state": self.rng.bit_generator.state,
            "exclude_list": self.exclude_list.copy(),
        }
        if self.prompt_stats_enabled:
            state["curriculum_state"] = {
                "prompt_pass_rates": self.prompt_pass_rates.copy(),
                "prompt_last_epoch": self.prompt_last_epoch.copy(),
                "prompt_last_step": self.prompt_last_step.copy(),
                "prompt_reuse_counts": self.prompt_reuse_counts.copy(),
                "active_replay_indices": list(self._active_replay_indices),
                "zero_pass_heap": self._zero_pass_heap.copy(),
                "zero_pass_lookup": self._zero_pass_lookup.copy(),
                "zero_pass_counter": self._zero_pass_counter,
                "zero_pass_epochs": {epoch: list(indices) for epoch, indices in self._zero_pass_epochs.items()},
                "permanent_exclusions": list(self._permanent_exclusions),
            }
        if self._progress_reorder_enabled:
            state["progress_reorder_state"] = {
                "pending_epoch_completions": list(self._pending_epoch_completions),
                "target_progress_unit": self._target_progress_unit,
                "last_reordered_progress_unit": self._last_reordered_progress_unit,
            }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore the iterator state from a checkpoint."""
        self.index = state["index"]
        self.epoch_number = state.get("epoch_number", 0)
        self.data = state["data"].copy()
        self.rng.bit_generator.state = state["rng_state"]
        self.exclude_list = state.get("exclude_list", [])
        curriculum_state = state.get("curriculum_state")
        if curriculum_state and self.prompt_stats_enabled:
            self.prompt_pass_rates = {
                int(k): float(v) for k, v in curriculum_state.get("prompt_pass_rates", {}).items()
            }
            self.prompt_last_epoch = {int(k): int(v) for k, v in curriculum_state.get("prompt_last_epoch", {}).items()}
            self.prompt_last_step = {int(k): int(v) for k, v in curriculum_state.get("prompt_last_step", {}).items()}
            self.prompt_reuse_counts = {
                int(k): int(v) for k, v in curriculum_state.get("prompt_reuse_counts", {}).items()
            }
            self._active_replay_indices = {int(v) for v in curriculum_state.get("active_replay_indices", [])}
            self._zero_pass_heap = [tuple(item) for item in curriculum_state.get("zero_pass_heap", [])]
            self._zero_pass_lookup = {
                int(k): (int(v[0]), int(v[1])) for k, v in curriculum_state.get("zero_pass_lookup", {}).items()
            }
            self._zero_pass_counter = int(curriculum_state.get("zero_pass_counter", self._zero_pass_counter))
            zero_pass_epochs = curriculum_state.get("zero_pass_epochs", {})
            self._zero_pass_epochs = defaultdict(set)
            for epoch, indices in zero_pass_epochs.items():
                self._zero_pass_epochs[int(epoch)] = {int(i) for i in indices}
            self._permanent_exclusions = {int(v) for v in curriculum_state.get("permanent_exclusions", [])}
            self._rebuild_replay_heap()
        if self._progress_reorder_enabled:
            progress_state = state.get("progress_reorder_state", {})
            pending_epochs = progress_state.get("pending_epoch_completions", [])
            self._pending_epoch_completions = deque(int(epoch) for epoch in pending_epochs)
            self._target_progress_unit = int(progress_state.get("target_progress_unit", self._target_progress_unit))
            self._last_reordered_progress_unit = int(
                progress_state.get("last_reordered_progress_unit", self._last_reordered_progress_unit)
            )
        else:
            self._pending_epoch_completions = deque()
            self._target_progress_unit = 0
            self._last_reordered_progress_unit = 0
        self._update_effective_size()

    def exclude_index(self, index: int) -> None:
        """Exclude provided data points from future sampling."""
        self.exclude_list.append(index)

    def update_prompt_pass_entries(self, prompt_pass_entries: list[PromptPassEntry]) -> None:
        """Update internal stats with the latest prompt pass information."""
        if not self.prompt_stats_enabled or not prompt_pass_entries:
            return

        for entry in prompt_pass_entries:
            dataset_index = int(entry.dataset_index)
            epoch_number = entry.epoch_number if entry.epoch_number is not None else self.epoch_number
            if epoch_number is None:
                continue
            epoch_number = int(epoch_number)
            self.prompt_pass_rates[dataset_index] = float(entry.pass_rate)
            self.prompt_last_epoch[dataset_index] = epoch_number
            if entry.training_step is not None:
                self.prompt_last_step[dataset_index] = int(entry.training_step)
            if entry.was_reused:
                self._active_replay_indices.discard(dataset_index)
            if entry.pass_rate == 0.0:
                self._record_zero_pass(dataset_index, epoch_number)
            else:
                self._remove_zero_pass(dataset_index)
                self._push_replay_candidate(dataset_index)

    def _record_zero_pass(self, dataset_index: int, epoch_number: int) -> None:
        self._zero_pass_epochs[epoch_number].add(dataset_index)
        insertion = (epoch_number, self._zero_pass_counter)
        self._zero_pass_counter += 1
        self._zero_pass_lookup[dataset_index] = insertion
        heapq.heappush(self._zero_pass_heap, (insertion[0], insertion[1], dataset_index))

    def _remove_zero_pass(self, dataset_index: int) -> None:
        self._zero_pass_lookup.pop(dataset_index, None)
        for zero_set in self._zero_pass_epochs.values():
            zero_set.discard(dataset_index)

    def _prepare_next_epoch(self, completed_epoch: int) -> None:
        if not self.curriculum_enabled:
            self._reset_epoch_order()
            return

        active_indices = self._active_indices()
        if not active_indices:
            self._set_epoch_data([])
            return

        zero_count_set = self._zero_pass_epochs.pop(completed_epoch, set()) if completed_epoch >= 0 else set()
        # Calculate quota based on total heap size (all accumulated zero-pass prompts)
        # This ensures fair selection across all epochs, not just the most recent one
        zero_quota = (
            math.ceil(self.zero_pass_fraction * len(self._zero_pass_heap))
            if len(self._zero_pass_heap) > 0 and self.zero_pass_fraction > 0
            else 0
        )

        active_set = set(active_indices)
        selected_zero = self._select_zero_pass_prompts(zero_quota, active_set)
        selected_zero_set = set(selected_zero)

        positives = []
        unknown = []
        for idx in active_indices:
            if idx in selected_zero_set:
                continue
            rate = self.prompt_pass_rates.get(idx)
            if rate is None:
                unknown.append(idx)
            elif rate > 0.0:
                positives.append((rate, idx))
            else:
                continue

        if self.curriculum_center_sort:
            positives.sort(key=lambda item: abs(item[0] - 0.5))
        else:
            positives.sort(key=lambda item: item[0], reverse=True)
        self.rng.shuffle(unknown)
        ordered = [idx for _, idx in positives] + unknown + selected_zero
        if not ordered:
            logger.warning("Curriculum ordering produced no prompts; falling back to random order.")
            self._reset_epoch_order()
            return
        self._set_epoch_data(ordered)

    def _active_indices(self) -> list[int]:
        return [int(idx) for idx in self.base_data if int(idx) not in self._permanent_exclusions]

    def _select_zero_pass_prompts(self, quota: int, active_set: set[int]) -> list[int]:
        if quota <= 0:
            return []

        selected: list[int] = []
        while len(selected) < quota and self._zero_pass_heap:
            epoch_number, insertion_order, dataset_index = heapq.heappop(self._zero_pass_heap)
            record = self._zero_pass_lookup.get(dataset_index)
            if record is None or record != (epoch_number, insertion_order):
                continue
            if dataset_index not in active_set:
                self._zero_pass_lookup.pop(dataset_index, None)
                continue
            selected.append(dataset_index)
            self._zero_pass_lookup.pop(dataset_index, None)
        return selected

    def _reset_epoch_order(self) -> None:
        active_indices = self._active_indices()
        if not active_indices:
            self._set_epoch_data([])
            return
        self.rng.shuffle(active_indices)
        self._set_epoch_data(active_indices)

    def _set_epoch_data(self, ordered_indices: list[int]) -> None:
        dtype = self.base_data.dtype if self.base_data.size > 0 else np.int64
        self.data = np.array(ordered_indices, dtype=dtype)
        self._update_effective_size()

    def _update_effective_size(self) -> None:
        """Ensure the effective dataset size is divisible by batch_size and filter out indices excluded in the last epoch"""
        if self.exclude_list:
            exclude_arr = np.array(self.exclude_list, dtype=self.data.dtype if self.data.size > 0 else np.int64)
            mask = ~np.isin(self.data, exclude_arr)
            self.data = self.data[mask]
            for idx in exclude_arr.tolist():
                int_idx = int(idx)
                self._permanent_exclusions.add(int_idx)
                self.prompt_pass_rates.pop(int_idx, None)
                self.prompt_last_epoch.pop(int_idx, None)
                self.prompt_last_step.pop(int_idx, None)
                self.prompt_reuse_counts.pop(int_idx, None)
                self._active_replay_indices.discard(int_idx)
                self._remove_zero_pass(int_idx)
            self.exclude_list = []

        self.effective_size = len(self.data) - (len(self.data) % self.batch_size)


@ray.remote(num_gpus=1)
class PolicyTrainerRayProcess(RayProcess):
    def from_pretrained(
        self,
        args: Args,
        model_config: ModelConfig,
        beaker_config: BeakerRuntimeConfig,
        wandb_url: str,
        tokenizer: PreTrainedTokenizer,
    ) -> int:
        # ------------------------------------------------------------
        # Monkey patch to load checkpoints with `weights_only=False`
        # otherwise it errors out with:
        # `_pickle.UnpicklingError: Weights only load failed. ` with pytorch 2.6.0
        from deepspeed.runtime.checkpoint_engine import torch_checkpoint_engine
        from deepspeed.utils import logger

        def load(self, path: str, map_location=None):
            logger.info(f"[Torch] Loading checkpoint from {path}...")
            partition = torch.load(path, map_location=map_location, weights_only=False)
            logger.info(f"[Torch] Loaded checkpoint from {path}.")
            return partition

        torch_checkpoint_engine.TorchCheckpointEngine.load = load

        # ------------------------------------------------------------
        self.args = args
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.beaker_config = beaker_config
        self.wandb_url = wandb_url
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(self.local_rank)
        self.node = socket.getfqdn()
        self.assigned_gpu_ids = utils.get_assigned_gpu_ids()

        # Set seeds for this worker (different per rank to avoid correlation)
        worker_seed = args.seed + self.local_rank
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

        deepspeed.init_distributed(timeout=timedelta(minutes=args.backend_timeout))

        ds_config = get_train_ds_config(
            offload=args.deepspeed_offload_param,
            adam_offload=args.deepspeed_offload_optimizer,
            stage=args.deepspeed_stage,
            bf16=True,
            zpg=args.deepspeed_zpg,
        )
        ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        ds_config["gradient_accumulation_steps"] = 1
        # @vwxyzjn: MAGIC: it's actually needed to initialize this `dschf`, so
        # https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
        # next line instructs transformers to partition the model directly over multiple gpus using
        # deepspeed.zero.Init when model's `from_pretrained` method is called.
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            dschf = HfDeepSpeedConfig(ds_config)
        else:
            dschf = None
        logger.info(f"Deepspeed config: {dschf=}")

        self.policy: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
            **({"device_map": {"": self.local_rank}} if args.deepspeed_stage != 3 else {}),
        )
        disable_dropout_in_model(self.policy)
        self.policy.gradient_checkpointing_enable()
        if args.set_weight_decay_on_bias_and_norm:
            optim_params = get_optimizer_grouped_parameters(self.policy, args.weight_decay)
        else:
            optim_params = self.policy.parameters()
        self.optimizer = torch.optim.AdamW(optim_params, lr=args.learning_rate, fused=args.fused_optimizer)
        num_scheduler_steps = args.num_training_steps * args.num_epochs * args.num_mini_batches
        warm_up_steps = args.warm_up_steps
        if args.warmup_ratio > 0.0:
            warm_up_steps = int(num_scheduler_steps * args.warmup_ratio)
        scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warm_up_steps,
            num_training_steps=num_scheduler_steps,
        )
        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.policy,
            optimizer=self.optimizer,
            config=ds_config,
            lr_scheduler=scheduler,
            dist_init_required=True,
        )
        optimization_steps_done = 0
        if args.checkpoint_state_dir:
            # check if the dir exists
            if not os.path.exists(args.checkpoint_state_dir):
                logger.warning(
                    f"Skipping loading checkpoint state from {args.checkpoint_state_dir} because it does not exist!"
                )
            else:
                path, states = self.model.load_checkpoint(
                    args.checkpoint_state_dir,
                    load_module_strict=True,
                    load_optimizer_states=True,
                    load_lr_scheduler_states=True,
                    load_module_only=False,
                )
                if path is None:
                    raise ValueError(f"Failed to load checkpoint from {args.checkpoint_state_dir}")
                optimization_steps_done = states["training_step"]

                rng_states = states["rng_states"]
                torch.set_rng_state(rng_states["torch_cpu_rng_state"])
                np.random.set_state(rng_states["numpy_rng_state"])
                random.setstate(rng_states["python_rng_state"])

                if torch.cuda.is_available() and "torch_cuda_rng_states" in rng_states:
                    # device_str, e.g. "cuda:0"
                    for device_str, rng_state in rng_states["torch_cuda_rng_states"].items():
                        device_id = int(device_str.split(":")[1])
                        torch.cuda.set_rng_state(rng_state, device_id)
                    if "torch_cuda_rng_state_all" in rng_states:
                        torch.cuda.set_rng_state_all(rng_states["torch_cuda_rng_state_all"])

                logger.info(f"{self.rank=}: Restored RNG states from checkpoint")

                # Save reference policy path to load later (after ref_policy is initialized)
                self.ref_policy_checkpoint_path = None
                if args.load_ref_policy and states.get("ref_policy_saved", False):
                    ref_policy_dir = os.path.join(args.checkpoint_state_dir, "ref_policy")
                    model_path = os.path.join(ref_policy_dir, "pytorch_model.bin")
                    if os.path.exists(model_path):
                        self.ref_policy_checkpoint_path = model_path
                        logger.info(f"{self.rank=}: Will load reference policy from {model_path}")

                logger.info(
                    f"{self.rank=}: Loaded checkpoint from {args.checkpoint_state_dir} with {optimization_steps_done=}"
                )
        self.model.train()

        # reference model
        if args.load_ref_policy:
            ds_config, self.ref_policy_hf_ds_config = get_eval_ds_config(
                offload=False,
                # inference model only has stage 3 (sharding) or stage 0 (no sharding)
                # stage 2 is optimizer sharding which doesn't apply to inference
                stage=args.deepspeed_stage if args.deepspeed_stage == 3 else 0,
                bf16=True,
                per_device_train_batch_size=args.per_device_train_batch_size,
            )

            self.ref_policy: PreTrainedModel = load_ref_policy(
                model_config=model_config,
                ds_config=ds_config,
                deepspeed_stage=args.deepspeed_stage,
                local_rank=self.local_rank,
                device=self.device,
                rank=self.rank,
                checkpoint_path=self.ref_policy_checkpoint_path
                if hasattr(self, "ref_policy_checkpoint_path")
                else None,
            )
        self.local_metrics = utils.MetricsTracker(device=self.device)
        return optimization_steps_done

    def forward(
        self,
        model: PreTrainedModel,
        query_response: torch.LongTensor,
        attention_mask: torch.LongTensor,
        position_ids: torch.LongTensor,
        pad_token_id: int,
        temperature: float,
        return_entropy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Replace pad tokens with 0s so that we don't run into index out of bounds errors
        padding_mask = query_response != pad_token_id
        input_ids = torch.masked_fill(query_response, ~padding_mask, 0)
        # NOTE: the [:-1] and [1:] are because the logits and generated tokens are off by 1 in index
        output = model(
            input_ids=input_ids[:, :-1],
            # @vwxyzjn: without clamp, we get index out of bounds errors; TODO: investigate
            attention_mask=attention_mask[:, :-1].clamp(0, 1),
            position_ids=position_ids[:, :-1],
            return_dict=True,
        )
        logits = output.logits
        logits /= temperature + 1e-7
        logprob = log_softmax_and_gather(logits, input_ids[:, 1:])

        # For now, entropy is just for monitoring, and we don't pass gradients through it.
        entropy = None
        if return_entropy:
            with torch.no_grad():
                entropy = entropy_from_logits(logits)

        return logprob, entropy

    def setup_model_update_group(self, vllm_engines):
        self.vllm_engines = vllm_engines
        if self.rank == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            vllm_num_engines, vllm_tensor_parallel_size = (
                self.args.vllm_num_engines,
                self.args.vllm_tensor_parallel_size,
            )
            world_size = vllm_num_engines * vllm_tensor_parallel_size + 1
            backend = self.args.vllm_sync_backend
            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * vllm_tensor_parallel_size + 1,
                    world_size,
                    "openrlhf",
                    backend=backend,
                    timeout_minutes=self.args.backend_timeout,
                )
                for i, engine in enumerate(vllm_engines)
            ]
            self.model_update_group = vllm_utils.init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name="openrlhf",
                timeout=timedelta(minutes=self.args.backend_timeout),
            )
            ray_get_with_progress(refs, desc="Initializing vLLM process groups", timeout=600)
        torch.distributed.barrier()

    def broadcast_to_vllm(self):
        # avoid OOM
        torch.cuda.empty_cache()
        model = self.model.module
        count, num_params = 0, len(list(model.named_parameters()))
        refss = []
        if self.args.gather_whole_model:
            with deepspeed.zero.GatheredParameters(model.parameters(), enabled=self.args.deepspeed_stage == 3):
                for name, param in model.named_parameters():
                    count += 1  # empty_cache at last param
                    # Fire all vllm engines for broadcast
                    if torch.distributed.get_rank() == 0:
                        shape = param.shape if self.args.deepspeed_stage != 3 else param.ds_shape
                        refs = [
                            engine.update_weight.remote(
                                name, dtype=str(param.dtype), shape=shape, empty_cache=count == num_params
                            )
                            for engine in self.vllm_engines
                        ]
                        refss.extend(refs)
                    if torch.distributed.get_rank() == 0:
                        torch.distributed.broadcast(param.data, 0, group=self.model_update_group)
        else:  # broadcast each parameter independently
            for name, param in model.named_parameters():
                count += 1
                if torch.distributed.get_rank() == 0:
                    shape = param.shape if self.args.deepspeed_stage != 3 else param.ds_shape
                    refs = [
                        engine.update_weight.remote(
                            name, dtype=str(param.dtype), shape=shape, empty_cache=count == num_params
                        )
                        for engine in self.vllm_engines
                    ]
                    refss.extend(refs)
                with deepspeed.zero.GatheredParameters([param], enabled=self.args.deepspeed_stage == 3):
                    if torch.distributed.get_rank() == 0:
                        torch.distributed.broadcast(param.data, 0, group=self.model_update_group)

        # Return futures instead of blocking - let caller handle completion
        all_refs = []
        if torch.distributed.get_rank() == 0:
            all_refs.extend(refss)
        return all_refs

    def update_ref_policy(self):
        if not self.args.load_ref_policy:
            return
        for ref_param, param in zip(self.ref_policy.parameters(), self.model.parameters()):
            if self.args.deepspeed_stage == 3:
                with deepspeed.zero.GatheredParameters([param, ref_param], modifier_rank=0):
                    if deepspeed.comm.get_rank() == 0:
                        ref_param.data.mul_(1.0 - self.args.alpha).add_(param.data, alpha=self.args.alpha)
            else:
                ref_param.data.mul_(1.0 - self.args.alpha).add_(param.data, alpha=self.args.alpha)

    def compute_logprobs(
        self,
        model: PreTrainedModel,
        collated_query_responses: list[torch.Tensor],
        collated_tool_masks: list[torch.Tensor],
        collated_attention_masks: list[torch.Tensor],
        collated_position_ids: list[torch.Tensor],
        collated_response_masks: list[torch.Tensor],
        pad_token_id: int,
        use_grad: bool = False,
        return_entropy: bool = False,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        collated_logprobs = []
        collated_entropies = [] if return_entropy else None

        context = contextlib.nullcontext() if use_grad else torch.no_grad()
        with context:
            for i in range(len(collated_query_responses)):
                query_response = collated_query_responses[i]
                tool_mask = collated_tool_masks[i]
                attention_mask = collated_attention_masks[i]
                position_id = collated_position_ids[i]
                response_mask = collated_response_masks[i]

                logprob, entropy = self.forward(
                    model,
                    query_response,
                    attention_mask,
                    position_id,
                    pad_token_id,
                    self.args.temperature,
                    return_entropy=return_entropy,
                )

                if self.args.mask_tool_use and self.args.tool_use:
                    response_mask = response_mask.bool() & tool_mask.bool()
                else:
                    response_mask = response_mask.bool()

                logprob = torch.masked_fill(logprob, ~response_mask[:, 1:], INVALID_LOGPROB)
                collated_logprobs.append(logprob)

                if return_entropy:
                    collated_entropies.append(entropy)

                torch.cuda.empty_cache()

        return collated_logprobs, collated_entropies

    def train(
        self,
        collated_query_responses,
        collated_tool_masks,
        collated_attention_masks,
        collated_position_ids,
        collated_advantages,
        collated_response_masks,
        collated_vllm_logprobs,
        pad_token_id: int,
        num_mini_batches: int,
    ):
        step_start_time = time.perf_counter()
        args = self.args
        to_device_inplace(collated_query_responses, self.device)
        to_device_inplace(collated_tool_masks, self.device)
        to_device_inplace(collated_attention_masks, self.device)
        to_device_inplace(collated_position_ids, self.device)
        to_device_inplace(collated_advantages, self.device)
        to_device_inplace(collated_response_masks, self.device)
        to_device_inplace(collated_vllm_logprobs, self.device)
        # accumulation steps should always be at least 1
        accumulation_steps = max(math.ceil(len(collated_query_responses) / num_mini_batches - 0.5), 1)
        leftover = len(collated_query_responses) % accumulation_steps
        if leftover > 0:
            collated_query_responses = collated_query_responses[0:-leftover]
            collated_tool_masks = collated_tool_masks[0:-leftover]
            collated_attention_masks = collated_attention_masks[0:-leftover]
            collated_position_ids = collated_position_ids[0:-leftover]
            collated_advantages = collated_advantages[0:-leftover]
            collated_response_masks = collated_response_masks[0:-leftover]
            collated_vllm_logprobs = collated_vllm_logprobs[0:-leftover]
            logger.warning(f"{leftover} samples are dropped due to batch size {num_mini_batches}")

        # recalculate the "real" number of mini-batches
        num_mini_batches = len(collated_query_responses) // accumulation_steps

        collated_ref_logprobs = []
        if args.load_ref_policy:
            with Timer("Inference Calculation", noop=self.rank != 0):
                collated_ref_logprobs, _ = self.compute_logprobs(
                    self.ref_policy,
                    collated_query_responses,
                    collated_tool_masks,
                    collated_attention_masks,
                    collated_position_ids,
                    collated_response_masks,
                    pad_token_id,
                    use_grad=False,
                    return_entropy=False,
                )
        # if we have multiple minibatches, we need to calculate the old logprobs for each minibatch
        # following gtrl scripts in just doing this on the current active policy, rather than use the logprobs
        # from the generator (note that async mode means these are a bit diff!)
        old_logprobs = [None for _ in range(len(collated_query_responses))]
        if num_mini_batches > 1:
            with Timer("Old logprobs Calculation", noop=self.rank != 0):
                local_old_logprobs = None
                if not args.use_vllm_logprobs:
                    local_old_logprobs, _ = self.compute_logprobs(
                        self.model,
                        collated_query_responses,
                        collated_tool_masks,
                        collated_attention_masks,
                        collated_position_ids,
                        collated_response_masks,
                        pad_token_id,
                        use_grad=False,
                        return_entropy=False,
                    )

                with torch.no_grad():
                    for i in range(len(collated_query_responses)):
                        tool_mask = collated_tool_masks[i]
                        response_mask = collated_response_masks[i]

                        if args.mask_tool_use and args.tool_use:
                            response_mask = response_mask.bool() & tool_mask.bool()
                        else:
                            response_mask = response_mask.bool()

                        vllm_old_logprob = collated_vllm_logprobs[i][:, 1:]
                        vllm_old_logprob = torch.masked_fill(vllm_old_logprob, ~response_mask[:, 1:], INVALID_LOGPROB)
                        vllm_old_logprob = torch.nan_to_num(vllm_old_logprob, nan=INVALID_LOGPROB)

                        if args.use_vllm_logprobs:
                            old_logprobs[i] = vllm_old_logprob
                        else:
                            old_logprobs[i] = local_old_logprobs[i]

                        torch.cuda.empty_cache()

        local_step = 0
        # Do multiple epochs of training on on-policy data (PPO-style), with a fresh random shuffle in each epoch
        with Timer("[Training Processes] Loss calculation", noop=self.rank != 0):
            kl1_stats = torch.zeros(len(collated_query_responses))
            kl2_stats = torch.zeros(len(collated_query_responses))
            kl3_stats = torch.zeros(len(collated_query_responses))
            kl4_stats = torch.zeros(len(collated_query_responses))
            kl_loss_stats = torch.zeros(len(collated_query_responses))
            pg_clipfrac_stats = torch.zeros(len(collated_query_responses))
            pg_loss_stats = torch.zeros(len(collated_query_responses))
            loss_stats = torch.zeros(len(collated_query_responses))
            ratio_stats = torch.zeros(len(collated_query_responses))
            entropy_stats = torch.zeros(len(collated_query_responses))
            advantage_stats = torch.zeros(len(collated_query_responses))
            for epoch_idx in range(args.num_epochs):
                for i in range(len(collated_query_responses)):
                    mb_query_responses = collated_query_responses[i]
                    mb_tool_mask = collated_tool_masks[i]
                    mb_advantages = collated_advantages[i]
                    mb_response_masks = collated_response_masks[i]
                    mb_response_masks_bool = mb_response_masks[:, 1:].bool()
                    # if masking snippets, do it here.
                    if args.mask_tool_use and args.tool_use:
                        mb_response_masks_bool = mb_response_masks[:, 1:].bool() & mb_tool_mask[:, 1:].bool()
                    mb_attention_mask = collated_attention_masks[i]
                    mb_position_id = collated_position_ids[i]
                    mb_local_logprobs, mb_entropy = self.forward(
                        self.model,
                        mb_query_responses,
                        mb_attention_mask,
                        mb_position_id,
                        pad_token_id,
                        args.temperature,
                        return_entropy=args.record_entropy,
                    )
                    mb_local_logprobs = torch.masked_fill(mb_local_logprobs, ~mb_response_masks_bool, INVALID_LOGPROB)
                    mb_vllm_logprobs = collated_vllm_logprobs[i][:, 1:]
                    mb_vllm_logprobs = torch.masked_fill(mb_vllm_logprobs, ~mb_response_masks_bool, INVALID_LOGPROB)
                    # Replace any remaining NaN values (query tokens in packed sequences are set to NaN by pack_sequences in rl_utils.py)
                    mb_vllm_logprobs = torch.nan_to_num(mb_vllm_logprobs, nan=INVALID_LOGPROB)

                    # Compare vLLM logprobs with local logprobs
                    with torch.no_grad():
                        valid_mask = mb_response_masks_bool & ~torch.isnan(mb_vllm_logprobs)
                        logprob_diff = (mb_local_logprobs - mb_vllm_logprobs).abs()
                        masked_diff = torch.masked_fill(logprob_diff, ~valid_mask, 0.0)
                        mean_diff = masked_diff.sum() / valid_mask.sum() if valid_mask.sum() > 0 else 0.0
                        max_diff = masked_diff.max()
                        std_diff = masked_diff[valid_mask].std() if valid_mask.sum() > 1 else 0.0

                        self.local_metrics["debug/vllm_vs_local_logprob_diff_mean"] = mean_diff.item()
                        self.local_metrics["debug/vllm_vs_local_logprob_diff_max"] = max_diff.item()
                        self.local_metrics["debug/vllm_vs_local_logprob_diff_std"] = std_diff.item()

                        reverse_kl = torch.exp(mb_vllm_logprobs) * (mb_vllm_logprobs - mb_local_logprobs)
                        masked_reverse_kl = torch.masked_fill(reverse_kl, ~valid_mask, 0.0)
                        mean_reverse_kl = masked_reverse_kl.sum() / valid_mask.sum() if valid_mask.sum() > 0 else 0.0
                        self.local_metrics["debug/vllm_local_reverse_kl"] = mean_reverse_kl.item()

                    mb_new_logprobs = mb_local_logprobs

                    # Cache the old logprobs
                    if num_mini_batches > 1:
                        mb_old_logprobs = old_logprobs[i]
                    else:
                        with torch.no_grad():
                            if epoch_idx == 0:
                                if args.use_vllm_logprobs:
                                    old_logprobs[i] = mb_vllm_logprobs
                                else:
                                    old_logprobs[i] = mb_local_logprobs.detach()
                            mb_old_logprobs = old_logprobs[i]

                    old_logprobs_mask = mb_old_logprobs != INVALID_LOGPROB
                    assert torch.all(old_logprobs_mask == mb_response_masks_bool), (
                        f"Old logprobs mask should match response mask. "
                        f"old_mask sum={old_logprobs_mask.sum()}, "
                        f"response_mask sum={mb_response_masks_bool.sum()}"
                    )

                    # Calculate the policy's loss
                    logprobs_diff = mb_new_logprobs - mb_old_logprobs
                    ratio = torch.exp(logprobs_diff)
                    pg_losses = -mb_advantages[:, 1:] * ratio
                    pg_losses2 = -mb_advantages[:, 1:] * torch.clamp(
                        ratio, 1.0 - args.clip_lower, 1.0 + args.clip_higher
                    )

                    # Apply truncated importance sampling if enabled
                    if args.truncated_importance_sampling_ratio_cap > 0 and mb_vllm_logprobs is not None:
                        old_logprobs_mask = mb_old_logprobs != INVALID_LOGPROB
                        vllm_logprobs_mask = mb_vllm_logprobs != INVALID_LOGPROB

                        assert torch.all(old_logprobs_mask == mb_response_masks_bool), (
                            f"Old logprobs mask should match response mask. "
                            f"old_mask sum={old_logprobs_mask.sum()}, "
                            f"response_mask sum={mb_response_masks_bool.sum()}"
                        )
                        assert torch.all(vllm_logprobs_mask == mb_response_masks_bool), (
                            f"vLLM logprobs mask should match response mask. "
                            f"vllm_mask sum={vllm_logprobs_mask.sum()}, "
                            f"response_mask sum={mb_response_masks_bool.sum()}"
                        )

                        valid_mask = mb_response_masks_bool

                        # Initialize importance ratio to 1.0 (no effect) for all positions
                        tis_imp_ratio = torch.ones_like(mb_old_logprobs)

                        if valid_mask.any():
                            # Calculate logprob difference only for valid positions
                            logprob_diff_is = mb_old_logprobs - mb_vllm_logprobs
                            # Clamp to prevent numerical overflow in exp
                            logprob_diff_is = torch.where(
                                valid_mask, logprob_diff_is.clamp(-10.0, 10.0), torch.zeros_like(logprob_diff_is)
                            )
                            # Compute importance ratio only for valid positions
                            tis_imp_ratio = torch.where(valid_mask, torch.exp(logprob_diff_is), tis_imp_ratio)
                            # Apply cap
                            tis_imp_ratio = torch.clamp(
                                tis_imp_ratio, max=args.truncated_importance_sampling_ratio_cap
                            )

                        # Apply importance sampling to losses
                        pg_losses = pg_losses * tis_imp_ratio
                        pg_losses2 = pg_losses2 * tis_imp_ratio

                    pg_loss_max = torch.max(pg_losses, pg_losses2)

                    if args.load_ref_policy:
                        mb_ref_logprob = collated_ref_logprobs[i]
                        # Here we recalculate kl: we want the KL loss to backpropagate through the model
                        # We also clamp the KL loss to avoid numerical instability
                        # https://chatgpt.com/share/679d0ed9-8f48-8011-926e-e274b15ae8ae
                        ref_logprobs_diff = (mb_new_logprobs - mb_ref_logprob).clamp(-40.0, 40.0)
                        kl1 = ref_logprobs_diff
                        kl2 = (ref_logprobs_diff) ** 2 / 2
                        kl3 = torch.expm1(-ref_logprobs_diff) + ref_logprobs_diff  # this is more numerically stable
                        kl4 = ratio * ref_logprobs_diff
                        if args.kl_estimator == "kl1":
                            kl = kl1
                        elif args.kl_estimator == "kl2":
                            kl = kl2
                        elif args.kl_estimator == "kl3":
                            kl = kl3
                        elif args.kl_estimator == "kl4":
                            kl = kl4
                        # grpo change: directly subtract KL in loss (add)
                        loss = masked_mean(
                            pg_loss_max + (args.beta * kl),
                            mb_response_masks_bool,
                            args.masked_mean_axis,
                            args.masked_mean_denominator,
                        )
                    else:
                        loss = masked_mean(
                            pg_loss_max, mb_response_masks_bool, args.masked_mean_axis, args.masked_mean_denominator
                        )
                    loss = loss / accumulation_steps
                    # Clear CUDA cache before backward pass to free memory for reduce_scatter operations
                    torch.cuda.empty_cache()
                    self.model.backward(loss)
                    if (local_step + 1) % accumulation_steps == 0:
                        self.model.step()
                    local_step += 1
                    with torch.no_grad():
                        if args.load_ref_policy:
                            # NOTE: in packed implementation, kl calculation are averages over response tokens
                            kl1_stats[i] = masked_mean(
                                kl1, mb_response_masks_bool, args.masked_mean_axis, args.masked_mean_denominator
                            ).float()
                            kl2_stats[i] = masked_mean(
                                kl2, mb_response_masks_bool, args.masked_mean_axis, args.masked_mean_denominator
                            ).float()
                            kl3_stats[i] = masked_mean(
                                kl3, mb_response_masks_bool, args.masked_mean_axis, args.masked_mean_denominator
                            ).float()
                            kl4_stats[i] = masked_mean(
                                kl4, mb_response_masks_bool, args.masked_mean_axis, args.masked_mean_denominator
                            ).float()
                            if args.kl_estimator == "kl1":
                                kl_loss_stats[i] = kl1_stats[i] * args.beta
                            elif args.kl_estimator == "kl2":
                                kl_loss_stats[i] = kl2_stats[i] * args.beta
                            elif args.kl_estimator == "kl3":
                                kl_loss_stats[i] = kl3_stats[i] * args.beta
                            elif args.kl_estimator == "kl4":
                                kl_loss_stats[i] = kl4_stats[i] * args.beta
                        pg_clipfrac_stats[i] = masked_mean(
                            (pg_losses2 > pg_losses).float(),
                            mb_response_masks_bool,
                            args.masked_mean_axis,
                            args.masked_mean_denominator,
                        )
                        pg_loss_stats[i] = masked_mean(
                            pg_loss_max, mb_response_masks_bool, args.masked_mean_axis, args.masked_mean_denominator
                        )
                        loss_stats[i] = loss
                        ratio_stats[i] = masked_mean(
                            ratio, mb_response_masks_bool, args.masked_mean_axis, args.masked_mean_denominator
                        )
                        if args.record_entropy:
                            # Calculate entropy statistics
                            entropy_stats[i] = masked_mean(
                                mb_entropy, mb_response_masks_bool, args.masked_mean_axis, args.masked_mean_denominator
                            ).float()
                        # Calculate absolute mean advantage statistics
                        advantage_stats[i] = masked_mean(
                            mb_advantages[:, 1:].abs(),
                            mb_response_masks_bool,
                            args.masked_mean_axis,
                            args.masked_mean_denominator,
                        )

            with torch.no_grad():
                if args.load_ref_policy:
                    self.local_metrics["objective/kl_avg"] = kl1_stats.mean()
                    self.local_metrics["objective/kl2_avg"] = kl2_stats.mean()
                    self.local_metrics["objective/kl3_avg"] = kl3_stats.mean()
                    self.local_metrics["objective/kl4_avg"] = kl4_stats.mean()
                    self.local_metrics["loss/kl_avg"] = kl_loss_stats.mean()
                self.local_metrics["loss/policy_avg"] = pg_loss_stats.mean()
                self.local_metrics["loss/total_avg"] = loss_stats.mean()
                self.local_metrics["policy/clipfrac_avg"] = pg_clipfrac_stats.mean()
                self.local_metrics["val/ratio"] = ratio_stats.mean()
                self.local_metrics["val/ratio_var"] = ratio_stats.var()
                self.local_metrics["val/advantage_abs_mean"] = advantage_stats.mean()
                if args.record_entropy:
                    self.local_metrics["policy/entropy_avg"] = entropy_stats.mean()
            self.local_metrics["lr"] = self.scheduler.get_last_lr()[0]
        # Lightweight per-learner GPU/utilization signals (unique keys to avoid aggregation).
        token_count = sum(t.numel() for t in collated_query_responses)
        step_time = max(time.perf_counter() - step_start_time, 1e-6)
        self.local_metrics[f"GPUs/learner_rank_{self.rank}/tokens_per_sec"] = token_count / step_time
        gpu_stats = utils.get_gpu_monitor_info(self.assigned_gpu_ids)
        util = gpu_stats["utilization"][0] if gpu_stats.get("utilization") else 0.0
        mem_used = gpu_stats["memory_used"][0] if gpu_stats.get("memory_used") else 0.0
        mem_total = gpu_stats["memory_total"][0] if gpu_stats.get("memory_total") else 1.0
        self.local_metrics[f"GPUs/learner_rank_{self.rank}/gpu_util_pct"] = util
        self.local_metrics[f"GPUs/learner_rank_{self.rank}/memory_used_gb"] = mem_used / 1e9
        self.local_metrics[f"GPUs/learner_rank_{self.rank}/memory_frac"] = mem_used / mem_total if mem_total else 0.0
        return self.local_metrics.get_metrics_list()

    def get_gpu_monitor_snapshot(self) -> dict[str, Any]:
        stats = utils.get_gpu_monitor_info(self.assigned_gpu_ids)
        return {
            "role": "learner",
            "rank": self.rank,
            "node": self.node,
            "gpu_ids": stats.get("gpu_ids"),
            "gpu_utilization": stats.get("utilization"),
            "gpu_memory_used": stats.get("memory_used"),
            "gpu_memory_total": stats.get("memory_total"),
        }

    def save_checkpoint_state(self, checkpoint_state_dir: str, client_state: dict[str, Any]) -> None:
        args = self.args

        # Save comprehensive RNG states for each rank
        rng_states = {
            "torch_cpu_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

        # Save CUDA RNG states for all devices
        if torch.cuda.is_available():
            rng_states["torch_cuda_rng_states"] = {
                f"cuda:{i}": torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())
            }
            rng_states["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        # Add RNG states to client_state
        client_state["rng_states"] = rng_states
        client_state["rank"] = self.rank

        # Save reference policy checkpoint (model only, no optimizer)
        if self.args.load_ref_policy:
            ref_policy_dir = os.path.join(checkpoint_state_dir, "ref_policy")
            os.makedirs(ref_policy_dir, exist_ok=True)

            # For reference policy, we save just the model weights
            # We can't use save_checkpoint because it would try to save DummyOptim
            # which doesn't have state_dict
            if self.rank == 0:
                # Only rank 0 saves the model state
                model_to_save = self.ref_policy.module if hasattr(self.ref_policy, "module") else self.ref_policy
                # Save the state dict
                torch.save(model_to_save.state_dict(), os.path.join(ref_policy_dir, "pytorch_model.bin"))
                logger.info(f"Saved reference policy model to {ref_policy_dir}")

            client_state["ref_policy_saved"] = True

        # Save the main model checkpoint with enhanced client state
        self.model.save_checkpoint(checkpoint_state_dir, client_state=client_state)

        # `save_checkpoint` needs to be called on all ranks, only rank 0 will have all the states
        if self.rank == 0:
            if args.keep_last_n_checkpoints >= 0:
                clean_last_n_checkpoints_deepspeed(checkpoint_state_dir, args.keep_last_n_checkpoints)

            # Sync to GCS if configured (check the actual target, not just gs_bucket_path)
            if args.gs_checkpoint_state_dir is not None:
                ray.remote(sync_gs_bucket).options(num_cpus=1).remote(
                    checkpoint_state_dir, args.gs_checkpoint_state_dir
                )

    def save_model(self, output_dir: str, chat_template_name: str, tokenizer: PreTrainedTokenizer) -> None:
        model_to_save = self.model
        if chat_template_name is not None and "olmo" in chat_template_name:
            # New chat template has no bos token, and two eos tokens: <|im_end|> and <|endoftext|>
            model_to_save.generation_config = get_olmo3_generation_config(tokenizer)

        if self.rank == 0:
            os.makedirs(output_dir, exist_ok=True)

        # save model weights for ZeRO2/3
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module

        # gather parameters
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            # only gather z3 params
            params_to_fetch = _z3_params_to_fetch([v])
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                vv = v.data.cpu()
                if self.rank == 0:
                    output_state_dict[k] = vv

        if self.rank == 0:
            state_dict = model_to_save.state_dict()

            # copy named_buffers with `persistent=True`
            for k, v in model_to_save.named_buffers():
                if k not in state_dict:
                    continue
                vv = v.data.cpu()
                output_state_dict[k] = vv

            state_dict_keys = set(state_dict.keys())
            output_state_dict_keys = set(output_state_dict.keys())

            # corner case for tie_word_embeddings, such as Qwen2-0.5B
            if getattr(model_to_save.config, "tie_word_embeddings", False) and "lm_head.weight" in state_dict_keys:
                state_dict_keys.remove("lm_head.weight")

            assert state_dict_keys.issubset(output_state_dict_keys), (
                f"mismatch keys {output_state_dict_keys.symmetric_difference(state_dict_keys)}"
            )

            # only save peft weights https://github.com/microsoft/DeepSpeed/issues/4295
            if isinstance(model_to_save, PeftModel):
                model_to_save.save_pretrained(output_dir)
                if self.stage == 3:
                    torch.save(
                        get_peft_model_state_dict(model_to_save, output_state_dict),
                        os.path.join(output_dir, "adapter_model.bin"),
                    )
            else:
                model_to_save.save_pretrained(output_dir, state_dict=output_state_dict)

            # save tokenizer
            self.tokenizer.save_pretrained(output_dir)

    # we need this because we don't know which node is rank 0 is on
    def launch_ai2_evals_on_weka_wrapper(self, step_dir, leaderboard_name, wandb_url, training_step):
        args = self.args
        if self.rank == 0:
            ray.remote(launch_ai2_evals_on_weka).options(num_cpus=1).remote(
                path=step_dir,
                leaderboard_name=leaderboard_name,
                oe_eval_max_length=args.oe_eval_max_length,
                wandb_url=wandb_url,
                training_step=training_step,
                oe_eval_tasks=args.oe_eval_tasks,
                stop_strings=args.stop_strings,
                gs_bucket_path=args.gs_bucket_path,
                eval_priority=args.eval_priority,
                eval_workspace=args.eval_workspace,
                beaker_image=args.oe_eval_beaker_image,
                oe_eval_gpu_multiplier=args.oe_eval_gpu_multiplier,
            )


class ModelGroup:
    def __init__(
        self, pg: PlacementGroup, ray_process_cls: RayProcess, num_gpus_per_node: list[int], single_gpu_mode: bool
    ):
        self.pg = pg
        self.ray_process_cls = ray_process_cls
        self.num_gpus_per_node = num_gpus_per_node
        self.num_gpus_per_actor = 0.48 if single_gpu_mode else 1
        self.num_cpus_per_actor = 4
        self.models = []
        world_size = sum(self.num_gpus_per_node)
        master_policy = ray_process_cls.options(
            num_cpus=self.num_cpus_per_actor,
            num_gpus=self.num_gpus_per_actor,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_bundle_index=0
            ),
        ).remote(world_size, 0, 0, None, None)

        self.models.append(master_policy)
        results, _ = ray_get_with_progress(
            [master_policy.get_master_addr_port.remote()], desc="Getting master address"
        )
        (master_addr, master_port) = results[0]

        def get_bundle_index(rank, num_gpus_per_node):
            """given a rank and a list of num_gpus_per_node, return the index of the bundle that the rank belongs to"""
            bundle_idx = 0
            while rank >= num_gpus_per_node[bundle_idx]:
                rank -= num_gpus_per_node[bundle_idx]
                bundle_idx += 1
            return bundle_idx

        assert get_bundle_index(0, [7, 8, 4]) == 0
        assert get_bundle_index(1, [7, 8, 4]) == 0
        assert get_bundle_index(7, [7, 8, 4]) == 1
        assert get_bundle_index(8, [7, 8, 4]) == 1
        assert get_bundle_index(9, [7, 8, 4]) == 1
        assert get_bundle_index(16, [7, 8, 4]) == 2

        # Setup worker models
        for rank in range(1, world_size):
            logger.debug(f"{rank=}, {world_size=}, {rank=}, {master_addr=}, {master_port=}")
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_bundle_index=get_bundle_index(rank, self.num_gpus_per_node)
            )
            worker_policy = ray_process_cls.options(
                num_cpus=self.num_cpus_per_actor,
                num_gpus=self.num_gpus_per_actor,
                scheduling_strategy=scheduling_strategy,
            ).remote(world_size, rank, 0, master_addr, master_port)
            self.models.append(worker_policy)

    def get_gpu_infos(self) -> list[dict[str, Any]]:
        infos, _ = ray_get_with_progress(
            [model.get_gpu_monitor_snapshot.remote() for model in self.models], desc="Collecting learner GPU info"
        )
        return infos


def log_gpu_assignments(
    policy_group: ModelGroup, vllm_engines: list[vllm_utils.LLMRayActor], with_tracking: bool
) -> None:
    """Log node/GPU placement for learners and engines to stdout and wandb."""
    learner_infos = policy_group.get_gpu_infos()
    engine_infos, _ = ray_get_with_progress(
        [engine.get_gpu_monitor_snapshot.remote() for engine in vllm_engines], desc="Collecting engine GPU info"
    )
    rows = []
    for info in learner_infos:
        rows.append(
            [
                "learner",
                info.get("rank"),
                info.get("node"),
                ",".join(map(str, info.get("gpu_ids") or [])),
                ",".join(map(str, info.get("gpu_utilization") or [])),
                ",".join(map(str, info.get("gpu_memory_used") or [])),
                ",".join(map(str, info.get("gpu_memory_total") or [])),
            ]
        )
    for info in engine_infos:
        rows.append(
            [
                "engine",
                info.get("engine_index"),
                info.get("node"),
                ",".join(map(str, info.get("gpu_ids") or [])),
                ",".join(map(str, info.get("gpu_utilization") or [])),
                ",".join(map(str, info.get("gpu_memory_used") or [])),
                ",".join(map(str, info.get("gpu_memory_total") or [])),
            ]
        )
    logger.info("GPU assignments (role, idx, node, gpu_ids, util_pct, mem_used, mem_total):")
    for row in rows:
        logger.info(row)

    if with_tracking:
        try:
            import wandb

            table = wandb.Table(
                columns=["role", "index", "node", "gpu_ids", "util_pct", "mem_used_bytes", "mem_total_bytes"],
                data=rows,
            )
            wandb.log({"GPUs/assignments": table}, step=0)
        except Exception as exc:
            logger.warning(f"Failed to log GPU assignments to wandb: {exc}")


def calculate_utilization_metrics(
    model_dims: utils.ModelDims,
    prompt_lengths: list[int],
    response_lengths: list[int],
    total_generation_time: float,
    samples_per_prompt: int,
    num_engines: int,
    num_gpus_per_engine: int,
    training_time: float,
    num_training_gpus: int,
) -> dict:
    """Calculate MFU and MBU metrics for model inference and training.

    Args:
        model_dims: Model dimensions with device information
        prompt_lengths: List of prompt lengths
        response_lengths: List of response lengths
        total_generation_time: Total time taken for generation (for actor metrics)
        samples_per_prompt: Number of samples generated per prompt
        num_engines: Number of vLLM engines for inference
        num_gpus_per_engine: Number of GPUs assigned to each vLLM engine (tensor parallel size)
        training_time: Time taken for training step (for learner metrics)
        num_training_gpus: Number of GPUs used for training (for learner metrics)

    Returns:
        Dict with the following keys:
            - actor_mfu: Model FLOPs utilization for inference (percentage)
            - actor_mbu: Model bandwidth utilization for inference (percentage)
            - learner_mfu: Model FLOPs utilization for training (percentage)
    """
    assert len(response_lengths) == len(prompt_lengths) * samples_per_prompt, (
        f"Expected {len(prompt_lengths) * samples_per_prompt} response lengths, got {len(response_lengths)}"
    )

    actor_metrics = model_dims.calculate_actor_utilization(
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        total_generation_time=total_generation_time,
        samples_per_prompt=samples_per_prompt,
        num_engines=num_engines,
        num_gpus_per_engine=num_gpus_per_engine,
    )

    learner_metrics = model_dims.calculate_learner_utilization(
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        training_time=training_time,
        samples_per_prompt=samples_per_prompt,
        num_training_gpus=num_training_gpus,
    )

    utilization_metrics = {f"actor_{k}": v for k, v in actor_metrics.items()}
    utilization_metrics["learner_mfu"] = learner_metrics["mfu"]

    return utilization_metrics


@dataclass
class BatchStatistics:
    prompt_lengths: list[int]
    response_lengths: list[int]
    filtered_prompts: int
    filtered_prompts_zero: int
    filtered_prompts_solved: int
    filtered_prompts_nonzero: int
    percent_solved_mean: float
    no_resampled_prompts: int
    total_prompts: int


@dataclass
class PromptPassEntry:
    prompt_id: str
    dataset_name: str
    dataset_index: int
    epoch_number: int | None
    training_step: int
    pass_rate: float
    was_reused: bool = False
    reuse_count: int | None = None
    cooldown_ready_step: int | None = None


@dataclass
class PromptReplayMetadata:
    """Metadata describing how a prompt entered the pipeline."""

    was_reused: bool = False
    reuse_count: int | None = None
    scheduled_step: int | None = None
    cooldown_ready_step: int | None = None

    @classmethod
    def empty(cls) -> PromptReplayMetadata:
        return cls()


@dataclass
class StepReplayState:
    """Tracks replay allocation for a single training step."""

    replay_budget: int
    issued_total: int = 0
    issued_replay: int = 0
    issued_new: int = 0
    reserved_indices: set[int] = field(default_factory=set)


class PromptPassTableLogger:
    """Tracks per-prompt pass rates, saves them locally, and logs to W&B."""

    def __init__(self, base_dir: str, run_name: str, enable_wandb_logging: bool):
        self.base_dir = os.path.abspath(base_dir)
        self.run_name = run_name
        self.enable_wandb_logging = enable_wandb_logging and wandb.run is not None
        self.run_dir = os.path.join(self.base_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.table_path = os.path.join(self.run_dir, "prompt_pass_rates.csv")
        self._rows: dict[str, dict[str, Any]] = {}

    def update(self, entries: list[PromptPassEntry]) -> None:
        for entry in entries:
            epoch_value = entry.epoch_number if entry.epoch_number is not None else -1
            column_name = f"epoch_{epoch_value}"
            row = self._rows.setdefault(
                entry.prompt_id, {"prompt_id": entry.prompt_id, "dataset_name": entry.dataset_name}
            )
            row[column_name] = entry.pass_rate

    def _build_dataframe(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=["prompt_id", "dataset_name"])
        df = pd.DataFrame(self._rows.values())
        epoch_columns = sorted(
            [col for col in df.columns if col.startswith("epoch_")], key=lambda name: int(name.split("_")[1])
        )
        ordered_cols = ["prompt_id", "dataset_name"] + epoch_columns
        return df.reindex(columns=ordered_cols).sort_values("prompt_id")

    def save_and_log(self, wandb_step: int | None = None) -> None:
        df = self._build_dataframe()
        if df.empty:
            return
        df.to_csv(self.table_path, index=False)
        if wandb_step is not None and self.enable_wandb_logging:
            wandb.log({"prompt_pass_rates": wandb.Table(dataframe=df)}, step=wandb_step)


class PromptReuseLogger:
    """Persists prompt replay metadata without disturbing the pass rate table."""

    def __init__(self, base_dir: str, run_name: str):
        self.base_dir = os.path.abspath(base_dir)
        self.run_dir = os.path.join(self.base_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.table_path = os.path.join(self.run_dir, "prompt_reuse_events.csv")
        self._rows: list[dict[str, Any]] = []

    def update(self, entries: list[PromptPassEntry]) -> None:
        for entry in entries:
            self._rows.append(
                {
                    "prompt_id": entry.prompt_id,
                    "dataset_name": entry.dataset_name,
                    "dataset_index": entry.dataset_index,
                    "training_step": entry.training_step,
                    "pass_rate": entry.pass_rate,
                    "was_reused": entry.was_reused,
                    "reuse_count": entry.reuse_count if entry.reuse_count is not None else 0,
                    "cooldown_ready_step": entry.cooldown_ready_step,
                }
            )

    def save(self) -> None:
        if not self._rows:
            return
        df = pd.DataFrame(self._rows)
        df.to_csv(self.table_path, index=False)


class PendingQueriesMap:
    """Thread-safe map for tracking pending queries with reference counting."""

    def __init__(self):
        self._map = {}  # dataset_idx -> dict entries
        self._lock = threading.Lock()

    def insert(
        self,
        dataset_idx,
        query,
        ground_truth,
        dataset,
        raw_query,
        dataset_source=None,
        metadata: PromptReplayMetadata | None = None,
    ):
        """Insert or increment count for a dataset index."""
        with self._lock:
            if dataset_idx in self._map:
                entry = self._map[dataset_idx]
                entry["count"] += 1
                entry["metadata"].append(metadata or PromptReplayMetadata.empty())
            else:
                self._map[dataset_idx] = {
                    "query": query,
                    "ground_truth": ground_truth,
                    "dataset": dataset,
                    "raw_query": raw_query,
                    "dataset_source": dataset_source,
                    "count": 1,
                    "metadata": deque([metadata or PromptReplayMetadata.empty()]),
                }

    def pop(self, dataset_idx):
        """Retrieve data and decrement count. Removes entry when count reaches 0."""
        with self._lock:
            if dataset_idx not in self._map:
                raise RuntimeError(f"Dataset index {dataset_idx} not found in pending_queries_map")

            entry = self._map[dataset_idx]
            query = entry["query"]
            ground_truth = entry["ground_truth"]
            dataset = entry["dataset"]
            raw_query = entry["raw_query"]
            dataset_source = entry.get("dataset_source")
            count = entry["count"]
            metadata_queue: deque[PromptReplayMetadata] = entry["metadata"]
            metadata = metadata_queue.popleft() if metadata_queue else PromptReplayMetadata.empty()

            if count > 1:
                entry["count"] -= 1
            else:
                del self._map[dataset_idx]

            return query, ground_truth, dataset, raw_query, dataset_source, metadata

    def __len__(self):
        """Return the number of entries in the map."""
        with self._lock:
            return len(self._map)

    def __contains__(self, dataset_idx):
        """Check if a dataset index is in the map."""
        with self._lock:
            return dataset_idx in self._map

    def __getitem__(self, dataset_idx):
        """Get the value for a dataset index."""
        with self._lock:
            return self._map[dataset_idx]

    def keys(self):
        """Return a view of the keys in the map."""
        with self._lock:
            return list(self._map.keys())


def accumulate_inference_batches(
    inference_results_Q: ray_queue.Queue,
    pending_queries_map: PendingQueriesMap,
    args: Args,
    generation_config: vllm.SamplingParams,
    num_prompts: int,
    model_dims: utils.ModelDims,
    tokenizer: PreTrainedTokenizer,
    reward_fn: Callable,
    actor_manager=None,
    timeout: float | None = None,
    active_sampling: bool = False,
    filter_zero_std_samples: bool = False,
    replenish_prompts: bool = False,
    no_resampling_pass_rate: float | None = None,
    iter_dataloader: ShufflingIterator | None = None,
    prompt_dataset: Dataset = None,
    param_prompt_Q: ray_queue.Queue | None = None,
    training_step: int = None,
    is_benchmark_mode: bool = False,
) -> tuple[GenerationResult, Batch, dict, BatchStatistics, list[PromptPassEntry]]:
    """Accumulate multiple inference results into a single training batch.

    Args:
        inference_results_Q: Queue containing individual GenerationResult objects (one per prompt)
        pending_queries_map: PendingQueriesMap instance for thread-safe query tracking
        args: Arguments containing vllm_num_engines and batch size info
        generation_config: Generation config containing n (number of samples per prompt)
        num_prompts: Number of prompts to accumulate
        timeout: Optional timeout in seconds for queue get operations. If None, blocks indefinitely.
        active_sampling: Whether to continue sampling until we have sampled num_prompts prompts with non-zero std
        filter_zero_std_samples: Whether to filter samples with zero reward std
        replenish_prompts: Add a prompt back onto the prompt_Q after receiving a finished result
        no_resampling_pass_rate: Optional rate at which to note samples solved at greater than this rate
            and exclude them from further sampling
        iter_dataloader: Optional, used for no_resampling_pass_rate
        param_prompt_Q: Queue containing prompts to send to generator, used to replenish used prompts
        is_benchmark_mode: If True, only process benchmark results; if False, only process eval results.
            Results that don't match are put back on the queue for later processing.

    Raises:
        queue.Empty: If timeout is specified and no data is available within timeout.

    Returns:
        Tuple of (combined_result, Batch with queries, ground_truths, datasets, prompt_lengths, response_lengths,
        prompt_pass_entries) or (ShutdownSentinel, None, None, None, None) if shutdown signal received
    """
    if no_resampling_pass_rate is not None:
        assert iter_dataloader is not None, "no_resampling requires the iter_dataloader passed"

    if replenish_prompts:
        assert param_prompt_Q is not None and iter_dataloader is not None and prompt_dataset is not None, (
            "replenish_prompts requires param_prompt_Q and iter_dataloader and prompt_dataset"
        )

    results = []
    all_queries = []
    all_ground_truths = []
    all_datasets = []
    all_dataset_sources = []
    all_raw_queries = []
    all_decoded_responses = []
    all_reward_metrics = []
    all_scores = []
    all_percent_solved = []
    prompt_pass_entries: list[PromptPassEntry] = []
    total_filtered_prompts = 0
    filtered_prompt_zero = 0
    filtered_prompt_solved = 0
    filtered_prompt_nonzero = 0
    total_no_resampled = 0
    progress_bar = tqdm(
        total=num_prompts,
        desc=f"Accumulating Responses and Rewarding {num_prompts} prompts",
        bar_format="{l_bar}{bar}{r_bar}\n",
        disable=not args.verbose,
    )
    num_prompts_sampled = 0
    engine_metrics_accumulator: dict[str, dict[str, Any]] = {}
    while num_prompts_sampled < num_prompts:
        result = inference_results_Q.get(timeout=timeout)

        if isinstance(result, ShutdownSentinel):
            return result, None, None, None, []

        # Filter results by benchmark mode: if result doesn't match expected mode, requeue it
        if result.is_benchmark != is_benchmark_mode:
            inference_results_Q.put(result)
            continue

        # Validate that each individual result has the expected number of responses
        assert len(result.responses) == generation_config.n, (
            f"Mismatch: individual prompt result has {len(result.responses)} responses "
            f"but expected {generation_config.n} samples per prompt. "
            f"Dataset index: {result.dataset_index}, Epoch: {result.epoch_number}"
        )

        query, ground_truth, dataset_name, raw_query, dataset_source, replay_metadata = pending_queries_map.pop(
            result.dataset_index
        )

        # Replenish generation queue with new prompt
        if replenish_prompts:
            replay_target_step = training_step + args.async_steps if training_step is not None else None
            dataset_index, prompt_metadata = iter_dataloader.next_for_step(replay_target_step)
            add_prompt_to_generator(
                prompt_dataset[dataset_index],
                dataset_index,
                iter_dataloader.epoch_number,
                training_step,
                pending_queries_map,
                param_prompt_Q,
                generation_config,
                is_eval=False,
                prompt_metadata=prompt_metadata,
            )

        # TODO(finbarrtimbers): Move this to LLMRayActor.
        for i in range(len(result.finish_reasons)):
            if result.finish_reasons[i] == "stop" and len(result.responses[i]) == 0:
                result.responses[i].append(tokenizer.eos_token_id)
                result.masks[i].append(1)
                result.logprobs[i].append(float("nan"))

        decoded_responses = tokenizer.batch_decode(result.responses, skip_special_tokens=True)

        # TODO(finbarrtimbers): Make PendingQueriesMap.pop return a Batch, and add a Batch.repeat method.
        k_queries = repeat_each([query], generation_config.n)
        k_ground_truths = repeat_each([ground_truth], generation_config.n)
        k_datasets = repeat_each([dataset_name], generation_config.n)
        k_dataset_sources = repeat_each([dataset_source], generation_config.n)
        k_raw_queries = repeat_each([raw_query], generation_config.n)

        scores, reward_metrics = asyncio.run(
            reward_fn(
                result.responses,
                decoded_responses,
                k_ground_truths,
                k_datasets,
                result.finish_reasons,
                result.request_info,
                k_raw_queries,
            )
        )

        percent_solved = np.mean(scores).item() / args.max_possible_score
        metadata = replay_metadata or PromptReplayMetadata.empty()
        if training_step is not None:
            prompt_pass_entries.append(
                PromptPassEntry(
                    prompt_id=f"{dataset_name}::{result.dataset_index}",
                    dataset_name=dataset_name,
                    dataset_index=result.dataset_index,
                    epoch_number=result.epoch_number,
                    training_step=training_step,
                    pass_rate=percent_solved,
                    was_reused=metadata.was_reused,
                    reuse_count=metadata.reuse_count,
                    cooldown_ready_step=metadata.cooldown_ready_step,
                )
            )
        # Don't resample prompt that was solved at more than no_resample_positive_rate
        if no_resampling_pass_rate is not None and percent_solved >= no_resampling_pass_rate:
            iter_dataloader.exclude_index(result.dataset_index)
            total_no_resampled += 1
            logging.debug(
                f"[Data Preparation Thread] Prompt solved at {percent_solved}, will be excluded from resampling, total no resampled: {total_no_resampled}"
            )

        # Filter out zero std prompts
        if filter_zero_std_samples and np.std(scores) == 0:
            # If we're not active sampling, still count this as a sample
            if not active_sampling:
                num_prompts_sampled += 1
                progress_bar.update(1)

            total_filtered_prompts += 1
            if scores[0] == 0:
                filtered_prompt_zero += 1
            elif scores[0] == args.max_possible_score:
                filtered_prompt_solved += 1
            else:
                filtered_prompt_nonzero += 1
            logging.debug(
                f"[Data Preparation Thread] Filtered prompt with reward std 0, total filtered {total_filtered_prompts}"
            )
            continue
        else:
            num_prompts_sampled += 1
            progress_bar.update(1)

        results.append(result)
        all_queries.extend(k_queries)
        all_ground_truths.extend(k_ground_truths)
        all_datasets.extend(k_datasets)
        all_dataset_sources.extend(k_dataset_sources)
        all_raw_queries.extend(k_raw_queries)
        all_decoded_responses.extend(decoded_responses)
        all_scores.extend(scores)
        all_reward_metrics.append(reward_metrics)
        all_percent_solved.append(percent_solved)
        engine_key = result.engine_id or (
            f"engine_{result.engine_index}" if result.engine_index is not None else "engine"
        )
        em = engine_metrics_accumulator.setdefault(
            engine_key,
            {
                "token_count": 0,
                "generation_time": 0.0,
                "prompt_lengths": [],
                "response_lengths": [],
                "gpu_utilization": [],
                "gpu_memory_used": [],
                "gpu_memory_total": [],
            },
        )
        if result.token_statistics:
            em["token_count"] += (
                result.token_statistics.num_prompt_tokens + result.token_statistics.num_response_tokens
            )
            em["generation_time"] += result.token_statistics.generation_time
        if result.prompt_lengths:
            # Store one prompt length per prompt; response lengths track per-sample outputs.
            em["prompt_lengths"].append(result.prompt_lengths[0])
        else:
            em["prompt_lengths"].append(len(query))
        if result.response_lengths:
            em["response_lengths"].extend(result.response_lengths)
        if result.gpu_utilization:
            em["gpu_utilization"].extend(result.gpu_utilization)
        if result.gpu_memory_used:
            em["gpu_memory_used"].extend(result.gpu_memory_used)
        if result.gpu_memory_total:
            em["gpu_memory_total"].extend(result.gpu_memory_total)

    # Combine all results into a single GenerationResult
    combined_responses = []
    combined_finish_reasons = []
    combined_masks = []
    combined_num_calls = []
    combined_timeouts = []
    combined_tool_errors = []
    combined_tool_outputs = []
    combined_tool_runtimes = []
    combined_tool_calleds = []
    combined_logprobs = []

    earliest_start_time = float("inf")
    prompt_lengths = []
    response_lengths = []

    total_prompt_tokens = 0
    total_response_tokens = 0
    max_generation_time = 0

    for i, result in enumerate(results):
        combined_responses.extend(result.responses)
        combined_finish_reasons.extend(result.finish_reasons)
        combined_masks.extend(result.masks)
        combined_num_calls.extend(result.request_info.num_calls)
        combined_timeouts.extend(result.request_info.timeouts)
        combined_tool_errors.extend(result.request_info.tool_errors)
        combined_tool_outputs.extend(result.request_info.tool_outputs)
        combined_tool_runtimes.extend(result.request_info.tool_runtimes)
        combined_tool_calleds.extend(result.request_info.tool_calleds)

        combined_logprobs.extend(result.logprobs)

        earliest_start_time = min(earliest_start_time, result.start_time)

        prompt_lengths.append(len(all_queries[i * generation_config.n]))

        for response in result.responses:
            response_lengths.append(len(response))

        total_prompt_tokens += result.token_statistics.num_prompt_tokens
        total_response_tokens += result.token_statistics.num_response_tokens
        max_generation_time = max(max_generation_time, result.token_statistics.generation_time)

    # Use the maximum generation time across engines since they work in parallel
    # This avoids including queue overhead and accumulation time in MFU/MBU calculations
    total_generation_time = max_generation_time

    accumulated_stats = TokenStatistics(
        num_prompt_tokens=total_prompt_tokens,
        num_response_tokens=total_response_tokens,
        generation_time=total_generation_time,
        earliest_start_time=earliest_start_time,
    )

    # Create combined RequestInfo
    combined_request_info = RequestInfo(
        num_calls=combined_num_calls,
        timeouts=combined_timeouts,
        tool_errors=combined_tool_errors,
        tool_outputs=combined_tool_outputs,
        tool_runtimes=combined_tool_runtimes,
        tool_calleds=combined_tool_calleds,
    )

    # Create combined GenerationResult
    combined_result = GenerationResult(
        responses=combined_responses,
        finish_reasons=combined_finish_reasons,
        masks=combined_masks,
        request_info=combined_request_info,
        dataset_index=None,
        epoch_number=results[0].epoch_number,
        token_statistics=accumulated_stats,
        logprobs=combined_logprobs,
    )

    if actor_manager is not None:
        ray.get(actor_manager.report_token_statistics.remote(accumulated_stats))

    # Note: We don't have dataset_indices here, but they're not needed for the returned batch
    batch = Batch(
        queries=all_queries,
        ground_truths=all_ground_truths,
        datasets=all_datasets,
        raw_queries=all_raw_queries,
        decoded_responses=all_decoded_responses,
        indices=None,  # Not meaningful for combined results
        scores=all_scores,
        dataset_sources=all_dataset_sources if all_dataset_sources else None,
    )

    combined_reward_metrics = combine_reward_metrics(all_reward_metrics)
    per_engine_metrics = {}
    for engine_id, data in engine_metrics_accumulator.items():
        prefix = f"GPUs/{engine_id}"
        total_time = max(data.get("generation_time", 0.0), 1e-6)
        tokens = data.get("token_count", 0)
        per_engine_metrics[f"{prefix}/tokens_per_sec"] = tokens / total_time
        if model_dims is not None and data.get("prompt_lengths") and data.get("response_lengths"):
            actor_util = model_dims.calculate_actor_utilization(
                prompt_lengths=data["prompt_lengths"],
                response_lengths=data["response_lengths"],
                total_generation_time=total_time,
                # Use the actual sampling config for this accumulation (eval/benchmark use n=1)
                samples_per_prompt=generation_config.n,
                num_engines=1,
                num_gpus_per_engine=args.vllm_tensor_parallel_size,
            )
            per_engine_metrics[f"{prefix}/actor_mfu"] = actor_util.get("mfu", 0.0)
            per_engine_metrics[f"{prefix}/actor_mbu"] = actor_util.get("mbu", 0.0)
        if data.get("gpu_utilization"):
            per_engine_metrics[f"{prefix}/gpu_util_pct"] = float(np.mean(data["gpu_utilization"]))
        if data.get("gpu_memory_used"):
            used_sum = float(np.sum(data["gpu_memory_used"]))
            total_sum = float(np.sum(data.get("gpu_memory_total", []))) if data.get("gpu_memory_total") else 0.0
            per_engine_metrics[f"{prefix}/memory_used_gb"] = used_sum / 1e9
            if total_sum > 0:
                per_engine_metrics[f"{prefix}/memory_frac"] = used_sum / total_sum
    combined_reward_metrics.update(per_engine_metrics)
    percent_solved_mean = np.mean(all_percent_solved) if all_percent_solved else 0.0

    batch_stats = BatchStatistics(
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        filtered_prompts=total_filtered_prompts,
        filtered_prompts_zero=filtered_prompt_zero,
        filtered_prompts_solved=filtered_prompt_solved,
        filtered_prompts_nonzero=filtered_prompt_nonzero,
        percent_solved_mean=percent_solved_mean,
        no_resampled_prompts=total_no_resampled,
        total_prompts=len(results),
    )
    logging.info(
        f"[Data Preparation Thread] Calculating rewards took {combined_reward_metrics['time/reward']} seconds"
    )

    return combined_result, batch, combined_reward_metrics, batch_stats, prompt_pass_entries


def data_preparation_thread(
    reward_fn: Callable,
    inference_results_Q: ray_queue.Queue,  # Ray queue
    param_prompt_Q: ray_queue.Queue,
    packed_sequences_Q: Queue,
    pending_queries_map: dict,
    args: Args,
    tokenizer: PreTrainedTokenizer,
    num_training_steps: int,
    generation_config,
    resume_training_step: int,
    iter_dataloader: ShufflingIterator,
    train_dataset: Dataset,
    actor_manager=None,
    model_dims: utils.ModelDims = None,
):
    for training_step in range(resume_training_step, num_training_steps + 1):
        # Streaming accumulation: collect results as they arrive
        with Timer("🚀 [Data Preparation Thread] Getting response ids") as timer:
            result, batch, reward_metrics, batch_stats, prompt_pass_entries = accumulate_inference_batches(
                inference_results_Q,
                pending_queries_map,
                args,
                generation_config,
                num_prompts=args.num_unique_prompts_rollout,
                model_dims=model_dims,
                tokenizer=tokenizer,
                reward_fn=reward_fn,
                actor_manager=actor_manager,
                active_sampling=args.active_sampling,
                filter_zero_std_samples=args.filter_zero_std_samples,
                replenish_prompts=True,
                no_resampling_pass_rate=args.no_resampling_pass_rate,
                iter_dataloader=iter_dataloader,
                prompt_dataset=train_dataset,
                param_prompt_Q=param_prompt_Q,
                training_step=training_step,
            )
            if isinstance(result, ShutdownSentinel):
                logger.info("[Data Preparation Thread] Received shutdown sentinel, exiting")
                return
            if (args.enable_prompt_pass_curriculum or args.enable_prompt_replay) and prompt_pass_entries:
                iter_dataloader.update_prompt_pass_entries(prompt_pass_entries)

        getting_response_time = timer.duration
        scores = np.array(batch.scores)

        good_outputs = [
            len(result.request_info.tool_outputs[i]) > 0
            and result.request_info.tool_calleds[i]
            and not result.request_info.timeouts[i]
            and not result.request_info.tool_errors[i]
            for i in range(len(result.request_info.tool_outputs))
        ]
        scores_per_prompt = scores.reshape(-1, args.num_samples_per_prompt_rollout)
        mean_grouped_rewards = scores_per_prompt.mean(axis=-1)
        mean_grouped_rewards = np.repeat(mean_grouped_rewards, args.num_samples_per_prompt_rollout, axis=0)
        std_grouped_rewards = scores_per_prompt.std(axis=-1)
        std_grouped_rewards = np.repeat(std_grouped_rewards, args.num_samples_per_prompt_rollout, axis=0)
        if args.advantage_normalization_type == "standard":
            advantages = (scores - mean_grouped_rewards) / (std_grouped_rewards + 1e-8)
        elif args.advantage_normalization_type == "centered":
            advantages = scores - mean_grouped_rewards
        else:
            raise ValueError(f"Invalid advantage normalization type: {args.advantage_normalization_type}")

        if args.mask_truncated_completions:
            stop_idxes = torch.tensor(
                [i for i in range(len(result.finish_reasons)) if result.finish_reasons[i] == "stop"]
            )
            num_truncated = len(result.finish_reasons) - len(stop_idxes)
            if num_truncated > 0:
                logger.info(
                    f"[Truncated completions filtering] Filtered {num_truncated} responses that didn't finish with 'stop'. "
                    f"Retention rate: {len(stop_idxes) / len(result.finish_reasons):.2%}"
                )
            scores = scores[stop_idxes]
            advantages = advantages[stop_idxes]
            batch = batch[stop_idxes.tolist()]
            result.responses = [result.responses[i] for i in stop_idxes]
            result.masks = [result.masks[i] for i in stop_idxes]
            result.finish_reasons = [result.finish_reasons[i] for i in stop_idxes]
            result.logprobs = [result.logprobs[i] for i in stop_idxes]

        with Timer("📦 [Data Preparation Thread] Packing sequences"):
            packed_sequences = pack_sequences(
                queries=batch.queries,
                responses=result.responses,
                masks=result.masks,
                pack_length=args.pack_length,
                pad_token_id=tokenizer.pad_token_id,
                vllm_logprobs=result.logprobs,
            )
            num_new_tokens = sum(len(seq) for seq in packed_sequences.query_responses)
            # Vectorized advantage calculation: create a lookup array where each index corresponds to a response mask value
            # and each value is the corresponding advantage score: index 0 is set to 0 since response masks start from 1 (1-indexed)
            lookup_advantages = np.zeros(len(advantages) + 1, dtype=np.float32)
            lookup_advantages[1:] = advantages
            packed_advantages = [
                torch.tensor(lookup_advantages[packed_mask], dtype=torch.float32)
                for packed_mask in packed_sequences.response_masks
            ]
            packed_sequences.advantages = packed_advantages

        # if we have less batches than world size, we need to pad out so each world is fine
        # ideally, you should avoid this since its wasting computation.
        if args.allow_world_padding:
            with Timer("🤺 [Data Preparation Thread] Padding sequences for world size"):
                shortfall = args.world_size - len(packed_sequences.query_responses)
                if shortfall > 0:
                    logger.warning(
                        f"Padding {shortfall} sequences for world size. In future, you should adjust your compute this."
                    )
                    # construct "dummy" sequences for padding out the world size
                    dummy_qr = torch.tensor([tokenizer.pad_token_id, tokenizer.eos_token_id], dtype=torch.long)
                    dummy_tool_mask = torch.zeros_like(dummy_qr)
                    dummy_attention = torch.tensor([1, 1], dtype=torch.long)
                    dummy_position_ids = torch.arange(len(dummy_qr), dtype=torch.long)
                    dummy_response_mask = torch.zeros_like(dummy_qr)
                    dummy_advantage = torch.zeros_like(dummy_qr, dtype=torch.float)
                    # pad out the world size
                    for _ in range(shortfall):
                        packed_sequences.query_responses.append(dummy_qr)
                        packed_sequences.tool_masks.append(dummy_tool_mask)
                        packed_sequences.attention_masks.append(dummy_attention)
                        packed_sequences.position_ids.append(dummy_position_ids)
                        packed_sequences.response_masks.append(dummy_response_mask)
                        packed_sequences.advantages.append(dummy_advantage)

        collated_data = prepare_collated_data_for_workers(
            packed_sequences, args.world_size, args.per_device_train_batch_size, tokenizer.pad_token_id
        )
        B = len(packed_sequences.query_responses) // args.world_size

        # Create a result package with metrics and data
        if len(result.responses) == 0:
            # Handle empty responses case
            # in this case, we won't log metrics, so it should be fine.
            metrics = {}
            logger.warning(f"No responses in batch {training_step}.")
        else:
            real_num_responses = len(result.responses)
            expected_num_responses = args.num_samples_per_prompt_rollout * args.num_unique_prompts_rollout

            unsolved_num_responses = (scores < args.max_possible_score).sum()
            sequence_lengths = np.array([len(response) for response in result.responses])
            sequence_length_solved = (
                np.array([]) if np.all(scores == 0) else np.array(sequence_lengths[scores == args.max_possible_score])
            )
            sequence_length_unsolved = (
                np.array([]) if np.all(scores == args.max_possible_score) else np.array(sequence_lengths[scores == 0])
            )
            stop_rate = sum(int(finish_reason == "stop") for finish_reason in result.finish_reasons) / len(
                result.finish_reasons
            )

            batch_metrics = asdict(batch_stats)
            batch_metrics_prefixed = {f"batch/{k}": v for k, v in batch_metrics.items()}

            metrics = {
                "scores": scores.mean(),
                "real_batch_size_ratio": real_num_responses / expected_num_responses,
                "unsolved_batch_size_ratio": unsolved_num_responses / real_num_responses,
                "packed_ratio": len(packed_sequences.query_responses) / real_num_responses,
                "val/solve_rate_hist": None,
                "val/total_reward_groups": real_num_responses / args.num_samples_per_prompt_rollout,
                "val/sequence_lengths": sequence_lengths.mean(),
                "val/sequence_lengths_min": sequence_lengths.min(),
                "val/sequence_lengths_max": sequence_lengths.max(),
                "val/sequence_lengths_unsolved": (
                    0 if len(sequence_length_unsolved) == 0 else sequence_length_unsolved.mean()
                ),
                "val/sequence_lengths_solved": (
                    0 if len(sequence_length_solved) == 0 else sequence_length_solved.mean()
                ),
                "val/sequence_lengths_unsolved_hist": sequence_length_unsolved,
                "val/sequence_lengths_solved_hist": sequence_length_solved,
                "val/stop_rate": stop_rate,
                "val/advantages_mean": advantages.mean(),
                "val/advantages_min": advantages.min(),
                "val/advantages_max": advantages.max(),
                "val/advantages_hist": advantages,
                "val/num_calls_rate": np.array(result.request_info.num_calls).mean(),
                "val/timeouts_rate": np.array(result.request_info.timeouts).mean(),
                "val/tool_errors_rate": np.array([len(item) > 0 for item in result.request_info.tool_errors]).mean(),
                "val/good_outputs_rate": np.array(good_outputs).mean(),
                "val/tool_runtimes_rate": np.array(result.request_info.tool_runtimes).mean(),
                "val/tool_calleds_rate": np.array(result.request_info.tool_calleds).mean(),
                "time/getting_response": getting_response_time,
                **reward_metrics,
                **batch_metrics_prefixed,
            }

            total_tokens = result.token_statistics.num_prompt_tokens + result.token_statistics.num_response_tokens
            metrics["val/actor_tokens_per_second"] = total_tokens / result.token_statistics.generation_time

        if args.save_traces:
            traces = {
                "scores": scores.tolist(),
                "finish_reasons": result.finish_reasons,
                "responses": result.responses,
                "training_step": training_step,
                **asdict(batch),  # Unpack all batch fields
                **reward_metrics,
            }
            os.makedirs(args.output_dir, exist_ok=True)
            with open(f"{args.output_dir}/traces_{args.run_name}.jsonl", "a") as f:
                json.dump(traces, f)
                f.write("\n")

        # Put the packed sequences and metrics into the output queue
        packed_sequences_Q.put(
            {
                "packed_sequences": packed_sequences,  # for debugging purposes
                "collated_data": collated_data,
                "metrics": metrics,
                "responses_count": len(result.responses),
                "num_new_tokens": num_new_tokens,
                "B": B,
                "prompt_lengths": batch_stats.prompt_lengths,
                "response_lengths": batch_stats.response_lengths,
                "num_filtered_prompts": batch_stats.filtered_prompts,
                "prompt_pass_entries": prompt_pass_entries,
            }
        )


def setup_runtime_variables(args: Args) -> Args:
    """Set up runtime variables for the experiment."""
    args.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if args.prompt_pass_table_dir is not None:
        args.prompt_pass_table_dir = os.path.abspath(args.prompt_pass_table_dir)
    if is_beaker_job():
        args.dataset_local_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
    args.world_size = sum(args.num_learners_per_node)
    args.num_training_steps = args.total_episodes // (
        args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
    )
    args.try_launch_beaker_eval_jobs_on_weka = args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job()
    if args.push_to_hub:
        if args.hf_repo_id is None:  # auto-generate one
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:  # first try to use AI2 entity
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:  # then try to use the user's entity
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
    if args.with_tracking and args.wandb_entity is None:
        args.wandb_entity = maybe_use_ai2_wandb_entity()
    args.tool_use = args.tools is not None and len(args.tools) > 0
    return args


def setup_experiment_tracking(args: Args, tc: TokenizerConfig, model_config: ModelConfig):
    """Setup experiment tracking and seeds."""
    all_configs = {}
    beaker_config = None
    if is_beaker_job():
        beaker_config = maybe_get_beaker_config()
        all_configs.update(vars(beaker_config))
    all_configs.update(**asdict(args), **asdict(tc), **asdict(model_config))

    wandb_url = None
    if args.with_tracking:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=all_configs,
            name=args.run_name,
            save_code=True,
            tags=[args.exp_name] + get_wandb_tags(),
        )
        wandb_url = wandb.run.get_url()
        maybe_update_beaker_description(wandb_url=wandb_url)

    return beaker_config, wandb_url


def _parse_dataset_mixer_pairs(dataset_mixer_list: list[str]) -> list[tuple[str, float]]:
    if len(dataset_mixer_list) == 0:
        raise ValueError("dataset_mixer_list is empty; cannot derive local eval subset")
    if len(dataset_mixer_list) % 2 != 0:
        raise ValueError(f"dataset_mixer_list must contain dataset/weight pairs, got: {dataset_mixer_list}")
    pairs: list[tuple[str, float]] = []
    for idx in range(0, len(dataset_mixer_list), 2):
        dataset_name = dataset_mixer_list[idx]
        weight_raw = dataset_mixer_list[idx + 1]
        try:
            weight_val = float(weight_raw)
        except ValueError as exc:
            raise ValueError(f"Could not parse dataset weight '{weight_raw}' in dataset_mixer_list") from exc
        pairs.append((dataset_name, weight_val))
    return pairs


def _distribute_counts(weights: list[float], total_count: int) -> list[int]:
    if total_count <= 0:
        raise ValueError("local_eval_subset_sample_count must be positive when deriving local eval subset")
    total_weight = sum(weights)
    if total_weight == 0:
        weights = [1.0] * len(weights)
        total_weight = float(len(weights))

    raw_counts = [total_count * (w / total_weight) for w in weights]
    counts = [math.floor(val) for val in raw_counts]
    remainders = [val - math.floor(val) for val in raw_counts]

    assigned = sum(counts)
    if assigned < total_count:
        order = sorted(range(len(counts)), key=lambda idx: remainders[idx], reverse=True)
        remaining = total_count - assigned
        idx = 0
        while remaining > 0 and order:
            target = order[idx % len(order)]
            counts[target] += 1
            remaining -= 1
            idx += 1
    elif assigned > total_count:
        order = sorted(range(len(counts)), key=lambda idx: remainders[idx])
        remaining = assigned - total_count
        idx = 0
        while remaining > 0 and order:
            target = order[idx % len(order)]
            if counts[target] > 0:
                counts[target] -= 1
                remaining -= 1
            idx += 1
    return counts


def maybe_configure_local_eval_subset(args: Args):
    """Override eval dataset mix using a subset of the training datasets."""
    if args.local_eval_subset_sample_count <= 0:
        return

    dataset_pairs = _parse_dataset_mixer_pairs(args.dataset_mixer_list)
    weights = [weight for _, weight in dataset_pairs]
    counts = _distribute_counts(weights, args.local_eval_subset_sample_count)

    eval_list: list[str] = []
    for (dataset_name, _), count in zip(dataset_pairs, counts):
        if count <= 0:
            continue
        eval_list.extend([dataset_name, str(count)])

    if not eval_list:
        raise ValueError(
            "Derived local eval subset is empty. Increase local_eval_subset_sample_count or check dataset mix."
        )

    args.dataset_mixer_eval_list = eval_list
    # eval_list contains (name, count) pairs, so num_datasets = len // 2
    args.dataset_mixer_eval_list_splits = ["train"] * (len(eval_list) // 2)

    logger.info(
        "Configured local eval subset using %d prompts across %d training datasets",
        args.local_eval_subset_sample_count,
        len(eval_list) // 2,
    )


def setup_datasets(args: Args, tc: TokenizerConfig, tokenizer: PreTrainedTokenizer):
    """Set up training, evaluation, and benchmark datasets."""
    system_prompt_override = None
    if args.system_prompt_override_file is not None:
        logger.info(f"Loading system prompt override from {args.system_prompt_override_file}")
        with open(args.system_prompt_override_file) as f:
            system_prompt_override = f.read().strip()
        logger.info(f"System prompt overriden to:\n#####\n{system_prompt_override}\n#####\n")

    transform_fn_args = [
        {"system_prompt_override": system_prompt_override},
        {"max_prompt_token_length": args.max_prompt_token_length},
    ]
    train_dataset = get_cached_dataset_tulu(
        dataset_mixer_list=args.dataset_mixer_list,
        dataset_mixer_list_splits=args.dataset_mixer_list_splits,
        tc=tc,
        dataset_transform_fn=args.dataset_transform_fn,
        transform_fn_args=transform_fn_args,
        dataset_cache_mode=args.dataset_cache_mode,
        dataset_config_hash=args.dataset_config_hash,
        hf_entity=args.hf_entity,
        dataset_local_cache_dir=args.dataset_local_cache_dir,
        dataset_skip_cache=args.dataset_skip_cache,
        system_prompt_override=system_prompt_override,
    )
    train_dataset = train_dataset.shuffle(seed=args.seed)

    eval_dataset = None
    if len(args.dataset_mixer_eval_list) > 0:
        eval_dataset = get_cached_dataset_tulu(
            dataset_mixer_list=args.dataset_mixer_eval_list,
            dataset_mixer_list_splits=args.dataset_mixer_eval_list_splits,
            tc=tc,
            dataset_transform_fn=args.dataset_transform_fn,
            transform_fn_args=transform_fn_args,
            hf_entity=args.hf_entity,
            dataset_cache_mode=args.dataset_cache_mode,
            dataset_config_hash=args.dataset_config_eval_hash,
            dataset_local_cache_dir=args.dataset_local_cache_dir,
            dataset_skip_cache=args.dataset_skip_cache,
            system_prompt_override=system_prompt_override,
        )
        if args.shuffle_eval_dataset:
            eval_dataset = eval_dataset.shuffle(seed=args.seed)

    benchmark_dataset = None
    if len(args.dataset_mixer_benchmark_list) > 0:
        logger.info(f"Loading benchmark dataset: {args.dataset_mixer_benchmark_list}")
        # Use benchmark-specific transform functions if provided, otherwise fall back to training transforms
        benchmark_transform_fn = (
            args.dataset_transform_fn_benchmark
            if args.dataset_transform_fn_benchmark is not None
            else args.dataset_transform_fn
        )
        logger.info(f"Using benchmark transform functions: {benchmark_transform_fn}")

        # Build transform_fn_args matching the benchmark transform functions
        # Each transform function needs a corresponding args dict
        benchmark_transform_fn_args = []
        for fn_name in benchmark_transform_fn:
            if fn_name == "rlvr_tokenize_v1":
                benchmark_transform_fn_args.append({"system_prompt_override": system_prompt_override})
            elif fn_name == "rlvr_max_length_filter_v1":
                benchmark_transform_fn_args.append({"max_prompt_token_length": args.max_prompt_token_length})
            else:
                # Converter functions and others don't need special args
                benchmark_transform_fn_args.append({})

        # Only keep the columns needed for benchmark evaluation
        # This ensures datasets with different schemas can be concatenated
        # NOTE: "messages" is needed as intermediate column between auto_convert and rlvr_tokenize
        benchmark_target_columns = [
            "messages",  # DEFAULT_SFT_MESSAGES_KEY (intermediate, used by rlvr_tokenize)
            "input_ids_prompt",  # INPUT_IDS_PROMPT_KEY
            "prompt",  # RAW_PROMPT_KEY
            "ground_truth",  # GROUND_TRUTHS_KEY
            "dataset",  # VERIFIER_SOURCE_KEY
            "dataset_source",  # DATASET_ORIGIN_KEY (added automatically if present)
        ]
        benchmark_dataset = get_cached_dataset_tulu(
            dataset_mixer_list=args.dataset_mixer_benchmark_list,
            dataset_mixer_list_splits=args.dataset_mixer_benchmark_list_splits,
            tc=tc,
            dataset_transform_fn=benchmark_transform_fn,
            transform_fn_args=benchmark_transform_fn_args,
            target_columns=benchmark_target_columns,
            hf_entity=args.hf_entity,
            dataset_cache_mode=args.dataset_cache_mode,
            dataset_config_hash=args.dataset_config_benchmark_hash,
            dataset_local_cache_dir=args.dataset_local_cache_dir,
            dataset_skip_cache=args.dataset_skip_cache,
            system_prompt_override=system_prompt_override,
            drop_dataset_source=False,  # Keep dataset_source for per-benchmark metrics
        )
        if args.shuffle_eval_dataset:
            benchmark_dataset = benchmark_dataset.shuffle(seed=args.seed)
        logger.info(f"Benchmark dataset loaded with {len(benchmark_dataset)} examples")
        logger.info(f"Benchmark dataset columns: {benchmark_dataset.column_names}")
        if "dataset_source" in benchmark_dataset.column_names:
            # Log unique dataset sources for per-benchmark metrics
            unique_sources = set(benchmark_dataset["dataset_source"])
            logger.info(f"Benchmark dataset sources: {unique_sources}")

    visualize_token(train_dataset[0][INPUT_IDS_PROMPT_KEY], tokenizer)

    return train_dataset, eval_dataset, benchmark_dataset


def create_model_and_optimizer(
    args: Args,
    tc: TokenizerConfig,
    model_config: ModelConfig,
    beaker_config: BeakerRuntimeConfig,
    wandb_url: str,
    tokenizer: PreTrainedTokenizer,
    inference_results_Q: ray_queue.Queue,
    param_prompt_Q: ray_queue.Queue,
    evaluation_inference_results_Q: ray_queue.Queue,
) -> tuple[ModelGroup, list[vllm_utils.LLMRayActor], dict, int, int]:
    """Create the model, optimizer, and vLLM engines."""
    # Create placement group
    bundles = [{"GPU": actor_num_gpus, "CPU": actor_num_gpus * 10} for actor_num_gpus in args.num_learners_per_node]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray_get_with_progress([pg.ready()], desc="Waiting for placement group")
    inits = []
    policy_group = ModelGroup(pg, PolicyTrainerRayProcess, args.num_learners_per_node, args.single_gpu_mode)
    wandb_url = wandb.run.get_url() if args.with_tracking else None
    inits.extend(
        model.from_pretrained.remote(args, model_config, beaker_config, wandb_url, tokenizer)
        for model in policy_group.models
    )

    # Set up tools
    max_len = args.max_prompt_token_length + args.response_length
    tool_objects = {}
    if args.tools:
        for tool in args.tools:
            if tool.lower() == "search":
                from open_instruct.search_utils.search_tool import SearchTool

                tool = SearchTool(
                    start_str="<query>",
                    end_str="</query>",
                    api_endpoint=args.search_api_endpoint,
                    number_documents_to_search=args.number_documents_to_search,
                )
                tool_objects[tool.end_str] = tool
                # Add tool end string to stop_strings
                args.stop_strings.append(tool.end_str)
            elif tool.lower() == "code":
                from open_instruct.tool_utils.tools import PythonCodeTool

                tool = PythonCodeTool(start_str="<code>", end_str="</code>", api_endpoint=args.code_tool_api_endpoint)
                tool_objects[tool.end_str] = tool
                # Add tool end string to stop_strings
                args.stop_strings.append(tool.end_str)
            else:
                raise ValueError(f"Unknown tool: {tool}")

    queues_to_monitor = {
        "Inference Results Queue": inference_results_Q,
        "Param Prompt Queue": param_prompt_Q,
        "Evaluation Queue": evaluation_inference_results_Q,
    }
    actor_manager = ray.remote(ActorManager).remote(queues_to_monitor, args)

    # Create vLLM engines with queues
    vllm_engines = vllm_utils.create_vllm_engines(
        args.vllm_num_engines,
        args.vllm_tensor_parallel_size,
        args.vllm_enforce_eager,
        tc.tokenizer_name_or_path,
        model_config.model_name_or_path,
        model_config.model_revision,
        args.seed,
        args.vllm_enable_prefix_caching,
        max_len,
        args.vllm_gpu_memory_utilization,
        args.single_gpu_mode,
        pg=pg if args.single_gpu_mode else None,
        tools=tool_objects,
        max_tool_calls=args.max_tool_calls,
        prompt_queue=param_prompt_Q,
        results_queue=inference_results_Q,
        eval_results_queue=evaluation_inference_results_Q,
        actor_manager=actor_manager,
        inflight_updates=args.inflight_updates,
    )

    results, _ = ray_get_with_progress(inits, desc="Initializing models")
    resume_training_step = results[0] + 1
    episode = (resume_training_step - 1) * args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
    logger.info("======== ✅ all models and vLLM engines initialized =========")

    kv_cache_max_concurrency = ray.get(vllm_engines[0].get_kv_cache_info.remote())
    ray.get(actor_manager.set_kv_cache_max_concurrency.remote(kv_cache_max_concurrency))
    expected_batch_size = (
        args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout // args.vllm_num_engines
    )
    if kv_cache_max_concurrency < expected_batch_size:
        nodes_needed = (
            args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout // kv_cache_max_concurrency
        )
        logger.warning(
            f"kv_cache_max_concurrency ({kv_cache_max_concurrency}) is lower than "
            f"num_unique_prompts_rollout * num_samples_per_prompt_rollout // vllm_num_engines ({expected_batch_size}). "
            f"This means actors will have to run multiple sequential batches, hurting performance. "
            f"You might want to use more inference nodes ({nodes_needed} nodes to generate the entire batch simultaneously)."
        )

    ray_get_with_progress(
        [m.setup_model_update_group.remote(vllm_engines=vllm_engines) for m in policy_group.models],
        desc="Setting up model update group",
    )
    logger.info("======== ✅ model update group setup successfully =========")

    return policy_group, vllm_engines, tool_objects, resume_training_step, episode, actor_manager


def create_generation_configs(args: Args):
    """Create generation configs for training and evaluation."""
    generation_config = vllm.SamplingParams(
        temperature=args.temperature,
        top_p=args.vllm_top_p,  # prevent rare out-of-vocab tokens with qwen
        max_tokens=args.response_length,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        n=args.num_samples_per_prompt_rollout,
        stop=args.stop_strings,
        seed=args.seed,
        logprobs=1,  # Enable logprobs to compare with local calculations
        # IMPORTANT: Set output_kind to FINAL_ONLY to ensure vLLM V1 properly handles n>1
        # With the default CUMULATIVE mode, vLLM V1 returns separate outputs for each
        # completion, making it difficult to aggregate them correctly. FINAL_ONLY mode
        # ensures all n completions are returned together in a single output.
        output_kind=vllm.sampling_params.RequestOutputKind.FINAL_ONLY,
    )
    eval_generation_config = generation_config.clone()
    eval_generation_config.temperature = 0.0
    eval_generation_config.n = 1
    benchmark_generation_config = generation_config.clone()
    benchmark_generation_config.temperature = 0.0
    benchmark_generation_config.n = 1
    return {"train": generation_config, "eval": eval_generation_config, "benchmark": benchmark_generation_config}


def add_prompt_to_generator(
    example: dict[str, Any],
    example_index: int,
    epoch_number: int,
    training_step: int,
    pending_queries_map: PendingQueriesMap,
    param_prompt_Q: ray_queue.Queue,
    generation_config,
    is_eval: bool,
    prompt_metadata: PromptReplayMetadata | None = None,
    is_benchmark: bool = False,
) -> None:
    """Split a batch into multiple inference batches and insert individual prompts into queues and mapping."""
    query = example[INPUT_IDS_PROMPT_KEY]
    ground_truth = example[GROUND_TRUTHS_KEY]
    dataset_name = example[VERIFIER_SOURCE_KEY]
    raw_query = example[RAW_PROMPT_KEY]
    dataset_source = example.get("dataset_source")  # Track which dataset this example came from
    pending_queries_map.insert(
        example_index, query, ground_truth, dataset_name, raw_query, dataset_source, prompt_metadata
    )

    param_prompt_Q.put(
        PromptRequest(
            prompt=query,
            generation_config=generation_config,
            epoch_number=epoch_number,
            training_step=training_step,
            dataset_index=example_index,
            is_eval=is_eval,
            is_benchmark=is_benchmark,
        )
    )


def load_data_from_packing_thread(
    packed_sequences_Q: Queue, num_total_tokens: int, stop_event: threading.Event, health_check_fn: Callable[[], None]
) -> tuple[
    list[dict[str, list[torch.Tensor]]] | None,
    dict[str, Any],
    int,
    int,
    list[int] | None,
    list[int] | None,
    int,
    list[PromptPassEntry],
]:
    """Get the packed sequences with advantages from the packing thread."""
    with Timer("[Main Thread] 📦 Getting packed sequences from thread") as timer:
        while True:
            if stop_event.is_set():
                logger.warning("[Main Thread] Stop event detected while waiting for packed sequences")
                return None, {}, num_total_tokens, 0, None, None, 0
            try:
                # When running at 32k generation length, it typically takes 900s to generate data,
                # so you might see this fire a bunch of times. That's normal!
                packed_data = packed_sequences_Q.get(timeout=300)
                break
            except Empty:
                health_check_fn()
                logger.warning("[Main Thread] Timeout waiting for packed sequences. Retrying...")
        data_thread_metrics = packed_data["metrics"]
        B = packed_data["B"]
        collated_data = packed_data["collated_data"]
        num_step_tokens = packed_data["num_new_tokens"]
        num_total_tokens += num_step_tokens
        prompt_lengths = packed_data["prompt_lengths"]
        response_lengths = packed_data["response_lengths"]
        num_filtered_prompts = packed_data["num_filtered_prompts"]
        prompt_pass_entries = packed_data.get("prompt_pass_entries", [])

    data_thread_metrics["time/trainer_idling"] = timer.duration
    if B == 0:
        logger.warning("[Main Thread] 🤡 After packing, there is not enough data to train")
        return None, data_thread_metrics, num_total_tokens, 0, None, None, 0, []
    return (
        collated_data,
        data_thread_metrics,
        num_total_tokens,
        num_step_tokens,
        prompt_lengths,
        response_lengths,
        num_filtered_prompts,
        prompt_pass_entries,
    )


def weight_sync_thread(
    args: Args,
    stop_event: threading.Event,
    weight_sync_trigger_event: threading.Event,
    policy_group: ModelGroup,
    actor_manager: ActorManager,
    weight_sync_metrics_Q: Queue,
    resume_training_step: int = 1,
):
    """Thread function that handles weight sync operations and actor manager coordination."""
    logger.info("[Weight Sync Thread] 🚀 Starting weight sync thread")
    if resume_training_step > 1:
        weight_sync_trigger_event.set()

    while not stop_event.is_set():
        # Wait for weight sync trigger from main thread
        if not weight_sync_trigger_event.wait(timeout=1.0):
            continue

        # Clear the event for next iteration
        weight_sync_trigger_event.clear()

        with Timer("[Weight Sync]") as timer:
            logger.debug("[Weight Sync Thread] Starting weight sync")

            # Set actors to stop
            ray.get(actor_manager.set_should_stop.remote(True))
            logger.debug("[Weight Sync Thread] Set should_stop to True for weight sync")

            # Broadcast weights to vLLM engines
            # First get the futures
            weight_broadcast_futures: list[ray.ObjectRef] = [m.broadcast_to_vllm.remote() for m in policy_group.models]

            # Wait for all weight updates to complete and collect individual timings
            _, actor_sync_times = ray_get_with_progress(
                weight_broadcast_futures,
                desc="[Weight Sync Thread] Waiting for weight updates to complete",
                enable=args.verbose,
            )

            # Allow actors to resume
            ray.get(actor_manager.set_should_stop.remote(False))
            logger.debug("[Weight Sync Thread] Set should_stop to False after weight sync")

        # Calculate distribution statistics
        sync_time_stats = {
            "time/weight_sync": timer.duration,
            "time/weight_sync_mean": np.mean(actor_sync_times),
            "time/weight_sync_min": np.min(actor_sync_times),
            "time/weight_sync_max": np.max(actor_sync_times),
            "time/weight_sync_median": np.median(actor_sync_times),
        }

        try:
            weight_sync_metrics_Q.put_nowait(sync_time_stats)
        except Full:
            logger.warning("[Weight Sync Thread] weight sync metrics queue full, skipping metric")

    logger.info("[Weight Sync Thread] 🛑 Stopping weight sync thread")


def one_training_step(
    args: Args,
    policy_group: ModelGroup,
    collated_data: list[dict[str, list[torch.Tensor]]],
    tokenizer: PreTrainedTokenizer,
    data_thread_metrics: dict[str, Any],
    episode: int,
    training_step: int,
    num_total_tokens: int,
    num_step_tokens: int,
    start_time: float,
    train_dataset: datasets.Dataset,
    training_start_time: float,
    wandb_url: str,
    chat_template_name: str,
    model_dims: utils.ModelDims,
    prompt_lengths: list[int],
    response_lengths: list[int],
    actor_manager: ActorManager | None = None,
    iter_dataloader: Iterator | None = None,
) -> None:
    """Train the model for one step."""
    update_ref_policy_future = []
    with Timer("[Main Thread] 🗡️ Training") as train_timer:
        metrics_list, _ = ray_get_with_progress(
            [
                policy_group.models[i].train.remote(
                    **collated_data[i], pad_token_id=tokenizer.pad_token_id, num_mini_batches=args.num_mini_batches
                )
                for i in range(args.world_size)
            ],
            desc=f"Running training step {training_step}",
        )
        if (
            args.load_ref_policy
            and args.ref_policy_update_freq is not None
            and training_step % args.ref_policy_update_freq == 0
            and args.alpha > 0
        ):
            update_ref_policy_future.extend(
                [policy_group.models[i].update_ref_policy.remote() for i in range(args.world_size)]
            )
            ray_get_with_progress(update_ref_policy_future, desc=f"Updating reference policy at step {training_step}")

    save_time = 0
    if args.save_freq > 0 and training_step % args.save_freq == 0 and (args.eval_on_step_0 or training_step > 1):
        with Timer("[Main Thread] 🗡️ Saving model") as timer:
            checkpoint_dir = f"{args.output_dir}_checkpoints"
            step_dir = os.path.join(checkpoint_dir, f"step_{training_step}")
            logger.info(f"Saving model at step {training_step} to {step_dir}")
            ray_get_with_progress(
                [
                    policy_group.models[i].save_model.remote(step_dir, chat_template_name, tokenizer)
                    for i in range(args.world_size)
                ],
                desc=f"Saving model at step {training_step}",
            )
            if args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job():
                leaderboard_name = f"{args.hf_repo_revision}_step_{training_step}"
                for i in range(args.world_size):
                    policy_group.models[i].launch_ai2_evals_on_weka_wrapper.remote(
                        step_dir, leaderboard_name, wandb_url, training_step
                    )
        save_time += timer.duration

    ray.get(actor_manager.report_training_step_time.remote(train_timer.duration))

    common_keys = set(metrics_list[0].keys())
    for m in metrics_list[1:]:
        common_keys &= set(m.keys())
    average_metrics = {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in common_keys}
    per_device_metrics = {}
    for m in metrics_list:
        for k, v in m.items():
            if k not in common_keys:
                per_device_metrics[k] = v
    step_time = time.perf_counter() - start_time
    total_training_time = time.perf_counter() - training_start_time

    total_generation_time = data_thread_metrics["time/getting_response"]

    utilization_metrics = calculate_utilization_metrics(
        model_dims=model_dims,
        prompt_lengths=prompt_lengths,
        response_lengths=response_lengths,
        total_generation_time=total_generation_time,
        samples_per_prompt=args.num_samples_per_prompt_rollout,
        num_engines=args.vllm_num_engines,
        num_gpus_per_engine=args.vllm_tensor_parallel_size,
        training_time=train_timer.duration,
        num_training_gpus=args.world_size,
    )

    metrics = {
        "episode": episode,
        "global_step": episode,
        "training_step": training_step,
        "val/num_total_tokens": num_total_tokens,
        "val/num_step_tokens": num_step_tokens,
        "epoch": episode / args.num_samples_per_prompt_rollout / len(train_dataset),
        "learner_tokens_per_second_overall": num_total_tokens / total_training_time,
        "learner_tokens_per_second_step": num_step_tokens / step_time,
        "time/total": step_time,
        "time/training": train_timer.duration,
        "time/saving": save_time,
        **data_thread_metrics,
        **average_metrics,
        **per_device_metrics,
        **utilization_metrics,
    }
    # Print only scalar metrics
    scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (float, int))}
    print_rich_single_line_metrics(scalar_metrics)

    if args.with_tracking:
        # Convert array/list metrics to wandb histograms for logging
        for key, value in metrics.items():
            if (isinstance(value, (np.ndarray, list))) and len(value) > 0:
                metrics[key] = wandb.Histogram(value)
        wandb.log(metrics, step=episode)


def maybe_evaluate(
    args: Args,
    training_step: int,
    evaluation_inference_results_Q: ray_queue.Queue,  # Ray queue
    tokenizer,
    reward_fn,
    episode,
    eval_pending_queries_map: PendingQueriesMap,
    eval_generation_config,
    generate_metrics_Q: Queue,
    num_eval_prompts: int,
    model_dims: utils.ModelDims,
    actor_manager=None,
):
    """Optionally evaluate the model."""
    try:
        # Allow configurable wait time so eval metrics can complete before logging.
        # Step 0 and final step need long timeout; mid-training can use shorter timeout
        # since results are generated async and may already be ready.
        eval_step = (training_step % args.local_eval_every == 0) or (training_step == 0 and args.eval_on_step_0)
        if eval_step:
            timeout = max(args.local_eval_timeout, 300.0)  # At least 5 minutes for initial/final eval
        else:
            timeout = 0.01

        # Accumulate evaluation results from all vLLM engines
        eval_result, eval_batch, eval_reward_metrics, _, _ = accumulate_inference_batches(
            evaluation_inference_results_Q,
            eval_pending_queries_map,
            args,
            eval_generation_config,
            num_prompts=num_eval_prompts,
            model_dims=model_dims,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            actor_manager=actor_manager,
            timeout=timeout,
            active_sampling=False,
            filter_zero_std_samples=False,
            replenish_prompts=False,
        )

        logger.info("[Main Thread] 📊 Evaluation responses received")

        eval_generate_metrics = {}
        try:
            eval_generate_metrics = generate_metrics_Q.get_nowait()
        except Empty:
            logger.info("[Main Thread] didn't get eval generation metrics")

        eval_sequence_lengths = np.array([len(response) for response in eval_result.responses])
        eval_stop_rate = sum(int(finish_reason == "stop") for finish_reason in eval_result.finish_reasons) / len(
            eval_result.finish_reasons
        )
        eval_reward_metrics = {f"eval/{key}": val for key, val in eval_reward_metrics.items()}
        eval_metrics = {
            "eval/scores": np.array(eval_batch.scores).mean(),
            "eval/sequence_lengths": eval_sequence_lengths.mean(),
            "eval/sequence_lengths_min": eval_sequence_lengths.min(),
            "eval/sequence_lengths_max": eval_sequence_lengths.max(),
            "eval/stop_rate": eval_stop_rate,
            **eval_reward_metrics,
        }
        if "time/generation" in eval_generate_metrics:
            eval_metrics["eval/generation_time"] = eval_generate_metrics["time/generation"]

        total_tokens = (
            eval_result.token_statistics.num_prompt_tokens + eval_result.token_statistics.num_response_tokens
        )
        eval_metrics["eval/actor_tokens_per_second"] = total_tokens / eval_result.token_statistics.generation_time

        print_rich_single_line_metrics(eval_metrics)

        table = {}
        table["prompt"] = tokenizer.batch_decode(eval_batch.queries if eval_batch else [])
        table["response"] = eval_batch.decoded_responses
        table["response"] = [item.replace(tokenizer.pad_token, "") for item in table["response"]]
        table["scores"] = eval_batch.scores
        table["ground_truth"] = eval_batch.ground_truths if eval_batch else []
        df = pd.DataFrame(table)

        if args.with_tracking:
            eval_metrics["sample_completions"] = wandb.Table(dataframe=df)
            wandb.log(eval_metrics, step=episode)
        else:
            print_rich_table(df.iloc[:1])
        del table
    except Empty:
        logger.warning("[Main Thread] 🙈 Evaluation responses not received")


def maybe_evaluate_benchmark(
    args: Args,
    training_step: int,
    evaluation_inference_results_Q: ray_queue.Queue,  # Ray queue (same as eval, run sequentially)
    tokenizer,
    reward_fn,
    episode,
    benchmark_pending_queries_map: PendingQueriesMap,
    benchmark_generation_config,
    generate_metrics_Q: Queue,
    num_benchmark_prompts: int,
    model_dims: utils.ModelDims,
    actor_manager=None,
):
    """Optionally evaluate the model on a separate benchmark dataset.

    Note: This uses the same evaluation_inference_results_Q as maybe_evaluate because
    both use is_eval=True and vLLM routes all eval prompts to the same queue. This is
    safe because eval and benchmark are run sequentially (eval completes before benchmark
    prompts are added).
    """
    if num_benchmark_prompts == 0:
        return
    try:
        # High timeout only when this is an actual benchmark step (prompts were added)
        benchmark_eval_freq = args.benchmark_eval_every if args.benchmark_eval_every >= 0 else args.local_eval_every
        is_benchmark_step = (training_step % benchmark_eval_freq == 0) or (training_step == 0 and args.eval_on_step_0)
        if is_benchmark_step:
            timeout = max(args.local_eval_timeout, 300.0)
        else:
            timeout = 0.01

        # Accumulate benchmark results from all vLLM engines
        benchmark_result, benchmark_batch, benchmark_reward_metrics, _, _ = accumulate_inference_batches(
            evaluation_inference_results_Q,
            benchmark_pending_queries_map,
            args,
            benchmark_generation_config,
            num_prompts=num_benchmark_prompts,
            model_dims=model_dims,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            actor_manager=actor_manager,
            timeout=timeout,
            active_sampling=False,
            filter_zero_std_samples=False,
            replenish_prompts=False,
            is_benchmark_mode=True,
        )

        logger.info("[Main Thread] 📊 Benchmark responses received")

        benchmark_generate_metrics = {}
        try:
            benchmark_generate_metrics = generate_metrics_Q.get_nowait()
        except Empty:
            logger.info("[Main Thread] didn't get benchmark generation metrics")

        benchmark_sequence_lengths = np.array([len(response) for response in benchmark_result.responses])
        benchmark_stop_rate = sum(
            int(finish_reason == "stop") for finish_reason in benchmark_result.finish_reasons
        ) / len(benchmark_result.finish_reasons)
        benchmark_reward_metrics = {f"benchmark/{key}": val for key, val in benchmark_reward_metrics.items()}

        # Compute overall (averaged) benchmark metrics
        benchmark_metrics = {
            "benchmark/scores": np.array(benchmark_batch.scores).mean(),
            "benchmark/sequence_lengths": benchmark_sequence_lengths.mean(),
            "benchmark/sequence_lengths_min": benchmark_sequence_lengths.min(),
            "benchmark/sequence_lengths_max": benchmark_sequence_lengths.max(),
            "benchmark/stop_rate": benchmark_stop_rate,
            **benchmark_reward_metrics,
        }
        if "time/generation" in benchmark_generate_metrics:
            benchmark_metrics["benchmark/generation_time"] = benchmark_generate_metrics["time/generation"]

        total_tokens = (
            benchmark_result.token_statistics.num_prompt_tokens + benchmark_result.token_statistics.num_response_tokens
        )
        benchmark_metrics["benchmark/actor_tokens_per_second"] = (
            total_tokens / benchmark_result.token_statistics.generation_time
        )

        # Compute per-benchmark metrics if dataset_sources is available
        if benchmark_batch.dataset_sources:
            # Map full dataset names to short display names
            SHORT_BENCHMARK_NAMES = {
                "HuggingFaceH4/MATH-500": "MATH500",
                "math-ai/minervamath": "MinervaMAth",
                "Hothan/OlympiadBench": "OlympiadBench",
                "math-ai/amc23": "AMC23",
                "mnoukhov/aime2024-25-rlvr": "AIME",
            }

            # Group results by dataset source
            source_to_indices = defaultdict(list)
            for idx, source in enumerate(benchmark_batch.dataset_sources):
                if source:  # Only include if source is not None
                    # Use short name if available, otherwise clean up the full name
                    # Strip config suffix (e.g., "Hothan/OlympiadBench:OE_TO_maths_en_COMP" -> "Hothan/OlympiadBench")
                    base_source = source.split(":")[0]
                    short_name = SHORT_BENCHMARK_NAMES.get(
                        base_source, source.replace("/", "_").replace(":", "_").replace("-", "_")
                    )
                    source_to_indices[short_name].append(idx)

            # Compute metrics for each benchmark dataset
            for source_name, indices in source_to_indices.items():
                source_scores = [benchmark_batch.scores[i] for i in indices]
                source_seq_lens = [benchmark_sequence_lengths[i] for i in indices]
                source_stop_count = sum(1 for i in indices if benchmark_result.finish_reasons[i] == "stop")

                # Add per-benchmark metrics
                benchmark_metrics[f"benchmark/{source_name}/scores"] = np.mean(source_scores)
                benchmark_metrics[f"benchmark/{source_name}/correct_rate"] = (
                    np.mean(source_scores) / args.max_possible_score
                )
                benchmark_metrics[f"benchmark/{source_name}/sequence_lengths"] = np.mean(source_seq_lens)
                benchmark_metrics[f"benchmark/{source_name}/stop_rate"] = source_stop_count / len(indices)
                benchmark_metrics[f"benchmark/{source_name}/count"] = len(indices)

        print_rich_single_line_metrics(benchmark_metrics)

        table = {}
        table["prompt"] = tokenizer.batch_decode(benchmark_batch.queries if benchmark_batch else [])
        table["response"] = benchmark_batch.decoded_responses
        table["response"] = [item.replace(tokenizer.pad_token, "") for item in table["response"]]
        table["scores"] = benchmark_batch.scores
        table["ground_truth"] = benchmark_batch.ground_truths if benchmark_batch else []
        if benchmark_batch.dataset_sources:
            table["dataset_source"] = benchmark_batch.dataset_sources

        # Debug: Log sample ground_truths for each benchmark to diagnose 0-score issues
        if benchmark_batch.dataset_sources:
            samples_by_source = defaultdict(list)
            for i, src in enumerate(benchmark_batch.dataset_sources):
                if src and len(samples_by_source[src]) < 2:  # Max 2 samples per benchmark
                    gt = benchmark_batch.ground_truths[i] if benchmark_batch.ground_truths else None
                    samples_by_source[src].append(
                        {
                            "ground_truth": gt,
                            "ground_truth_type": type(gt).__name__,
                            "score": benchmark_batch.scores[i] if benchmark_batch.scores else None,
                            "response_snippet": table["response"][i][:300] if table["response"] else None,
                        }
                    )
            for src, samples in samples_by_source.items():
                logger.info(f"[DEBUG] Benchmark '{src}' samples: {samples}")

        df = pd.DataFrame(table)

        if args.with_tracking:
            benchmark_metrics["benchmark_sample_completions"] = wandb.Table(dataframe=df)
            wandb.log(benchmark_metrics, step=episode)
        else:
            print_rich_table(df.iloc[:1])
        del table
    except Empty:
        logger.warning("[Main Thread] 🙈 Benchmark responses not received")


def save_final_model(
    args: Args,
    policy_group: ModelGroup,
    tokenizer: PreTrainedTokenizer,
    training_step: int,
    wandb_url: str,
    chat_template_name: str,
):
    """Save the final model and launch evaluation jobs if configured."""
    logger.info(f"Saving final model at step {training_step} to {args.output_dir}")
    with Timer("[Main Thread] 🗡️ Saving model"):
        ray_get_with_progress(
            [
                policy_group.models[i].save_model.remote(args.output_dir, chat_template_name, tokenizer)
                for i in range(args.world_size)
            ],
            desc="Saving final model",
        )
        if args.try_launch_beaker_eval_jobs_on_weka and is_beaker_job():
            leaderboard_name = args.hf_repo_revision
            for i in range(args.world_size):
                policy_group.models[i].launch_ai2_evals_on_weka_wrapper.remote(
                    args.output_dir, leaderboard_name, wandb_url, training_step
                )


def make_tokenizer(tc: TokenizerConfig, model_config: ModelConfig):
    """Setup tokenizer with appropriate configuration."""
    tc.tokenizer_revision = model_config.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision
    tc.tokenizer_name_or_path = (
        model_config.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    if (
        tc.tokenizer_revision != model_config.model_revision
        and tc.tokenizer_name_or_path != model_config.model_name_or_path
    ):
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tc.tokenizer_revision=}` is different
                   from the model revision `{model_config.model_revision=}` or the tokenizer name `{tc.tokenizer_name_or_path=}`
                   is different from the model name `{model_config.model_name_or_path=}`."""
        logger.warning(warning)
    return tc.tokenizer


def make_reward_fn(args: Args) -> Callable:
    """Create a reward function based on the provided arguments."""
    reward_fn_mapping = build_all_verifiers(args)

    async def reward_fn(
        responses: list[torch.Tensor],
        decoded_responses: list[str],
        ground_truths: list[Any],
        datasets: list[str],
        finish_reasons: list[str],
        infos: list[list[int]],
        queries: list[str] | None = None,
    ) -> (list[float], dict[str, Any]):
        timeouts = infos.timeouts
        tool_errors = infos.tool_errors
        tool_outputs = infos.tool_outputs
        tool_calleds = infos.tool_calleds
        good_outputs = [
            len(tool_outputs[i]) > 0 and tool_calleds[i] and not timeouts[i] and not tool_errors[i]
            for i in range(len(tool_outputs))
        ]
        scores = [0] * len(decoded_responses)
        metrics = {}

        reward_time = 0
        if args.apply_r1_style_format_reward:
            with Timer(
                "[Data Preparation Thread] Calculating rewards -- 🧮 Calculating format reward", noop=True
            ) as timer:
                format_scores = soft_format_reward_func(decoded_responses, args.r1_style_format_reward)
                if len(format_scores) != len(scores):
                    raise ValueError(f"{len(format_scores)=} != {len(scores)=}")
                for i in range(len(format_scores)):
                    scores[i] = format_scores[i] + scores[i]
                metrics["val/format_scores"] = np.array(format_scores).mean()

            reward_time += timer.duration

        if args.apply_verifiable_reward:
            with Timer(
                "[Data Preparation Thread] Calculating rewards -- 🏆 Applying verifiable reward", noop=True
            ) as timer:
                verifiable_rewards, per_func_rewards = await apply_verifiable_reward(
                    reward_fn_mapping,
                    responses,
                    decoded_responses,
                    ground_truths,
                    datasets,
                    reward_mult=args.verification_reward,
                    queries=queries,
                )
                if len(verifiable_rewards) != len(scores):
                    raise ValueError(f"{len(verifiable_rewards)=} != {len(scores)=}")
                # slightly complex combo of good outputs and additive format reward
                for i in range(len(verifiable_rewards)):
                    if not args.only_reward_good_outputs or (good_outputs[i] and args.only_reward_good_outputs):
                        if args.apply_r1_style_format_reward and args.additive_format_reward:
                            scores[i] = verifiable_rewards[i] + scores[i]
                        elif args.apply_r1_style_format_reward and not args.additive_format_reward:
                            scores[i] = verifiable_rewards[i] if format_scores[i] == 1 else 0
                        else:
                            scores[i] = verifiable_rewards[i]
                np_verifiable_rewards = np.array(verifiable_rewards)
                metrics["objective/verifiable_reward"] = np_verifiable_rewards.mean()
                metrics["objective/verifiable_correct_rate"] = (np_verifiable_rewards > 0.0).mean()
                # reshuffle around per_func rewards
                per_func_lists = defaultdict(list)
                for reward_dict in per_func_rewards:
                    for key, value in reward_dict.items():
                        per_func_lists[key].append(value)
                # log per function rewards
                for key, value in per_func_lists.items():
                    np_value = np.array(value)
                    metrics[f"objective/{key}_reward"] = np_value.mean()
                    metrics[f"objective/{key}_correct_rate"] = (np_value > 0.0).mean()

            reward_time += timer.duration

        # this gets applied at the very end since it replaces (rather than adds to) the existing reward.
        if args.non_stop_penalty:
            with Timer(
                "[Data Preparation Thread] Calculating rewards -- 🦖 Applying non stop penalty", noop=True
            ) as timer:
                assert len(finish_reasons) == len(scores)
                for i in range(len(finish_reasons)):
                    if finish_reasons[i] != "stop":
                        scores[i] = args.non_stop_penalty_value

            reward_time += timer.duration

        metrics["time/reward"] = reward_time

        return scores, metrics

    return reward_fn


def cleanup_judge_clients():
    """Cleans up all LLM judge clients."""
    asyncio.run(cleanup_all_llm_judge_clients())
    logger.info("✅ LLM judge clients cleaned up")


def cleanup_training_resources(
    stop_event: threading.Event,
    executor: futures.ThreadPoolExecutor,
    queues: list[ray_queue.Queue],
    actor_manager: ActorManager,
) -> None:
    """Clean up all training resources including threads and Ray queues."""
    stop_event.set()

    logger.info("Signaling all actors to stop...")
    ray.get(actor_manager.set_should_stop.remote(True))
    logger.info("✅ Signaled all actors to stop")

    # Clean up ActorManager resources
    logger.info("Cleaning up ActorManager resources...")
    ray.get(actor_manager.cleanup.remote())
    logger.info("✅ ActorManager resources cleaned up")

    logger.info("Pushing shutdown sentinel to queues...")
    # Push sentinel to the first queue (inference_results_Q)
    if queues and len(queues) > 0:
        queues[0].put(ShutdownSentinel(), timeout=1)

    logger.info("Shutting down Ray queues...")
    if queues and len(queues) > 0:
        [queue.shutdown() for queue in queues]
    logger.info("Shutting down thread pool executor...")
    executor.shutdown(wait=True)

    # Clean up judge clients
    cleanup_judge_clients()

    # Shutdown Ray only from the main process (rank 0) or when DDP isn't initialized
    try:
        is_ddp = dist.is_available() and dist.is_initialized()
        is_rank0 = (not is_ddp) or (dist.get_rank() == 0)
        if is_rank0 and ray.is_initialized():
            logger.info("Shutting down Ray...")
            ray.shutdown()
            logger.info("✅ Ray shut down")
    except Exception as e:
        logger.warning(f"Ray shutdown failed: {e}")

    # Clean up distributed process group if it was initialized
    if dist.is_initialized():
        logger.info("Destroying process group...")
        dist.destroy_process_group()
        logger.info("✅ Process group destroyed")


def run_training(
    args,
    tokenizer,
    train_dataset,
    eval_dataset,
    benchmark_dataset,
    policy_group,
    vllm_engines,
    generation_configs,
    iter_dataloader,
    reward_fn,
    resume_training_step,
    episode,
    wandb_url,
    tc,
    stop_event,
    executor,
    inference_results_Q,
    param_prompt_Q,
    evaluation_inference_results_Q,
    packed_sequences_Q,
    pending_queries_map,
    eval_pending_queries_map,
    benchmark_pending_queries_map,
    generate_metrics_Q,
    weight_sync_metrics_Q,
    actor_manager: ActorManager,
    model_dims: utils.ModelDims,
    checkpoint_state=None,
):
    if resume_training_step > 1:
        logger.info(f"[Main Thread] Resuming training from step {resume_training_step}")

    logger.info("======== ✅ weight sync thread starts =========")
    weight_sync_trigger_event = threading.Event()
    weight_sync_thread_future = executor.submit(
        weight_sync_thread,
        args,
        stop_event,
        weight_sync_trigger_event,
        policy_group,
        actor_manager,
        weight_sync_metrics_Q,
        resume_training_step,
    )

    """Run the main training loop with worker threads."""
    ray_get_with_progress(
        [engine.ready.remote() for engine in vllm_engines], "Checking engines are ready to work", timeout=300
    )

    logger.info("======== ✅ data preparation thread starts =========")
    packing_future = executor.submit(
        data_preparation_thread,
        reward_fn,
        inference_results_Q,
        param_prompt_Q,
        packed_sequences_Q,
        pending_queries_map,
        args,
        tokenizer,
        args.num_training_steps,
        generation_configs["train"],
        resume_training_step,
        iter_dataloader,
        train_dataset,
        actor_manager,
        model_dims,
    )

    prompt_pass_logger = None
    prompt_reuse_logger = None
    if args.prompt_pass_table_dir:
        prompt_pass_logger = PromptPassTableLogger(
            base_dir=args.prompt_pass_table_dir, run_name=args.run_name, enable_wandb_logging=args.with_tracking
        )
        prompt_reuse_logger = PromptReuseLogger(base_dir=args.prompt_pass_table_dir, run_name=args.run_name)

    def health_check_fn():
        [f.result() for f in [packing_future, weight_sync_thread_future] if f.done()]
        ray_get_with_progress(
            [engine.check_background_threads.remote() for engine in vllm_engines],
            desc="Checking vLLM engine health",
            enable=False,
        )

    # Send initial data to ensure we have a N-step offset.
    for prefill_idx in range(args.async_steps * args.num_unique_prompts_rollout):
        prefill_target_step = resume_training_step + (prefill_idx // args.num_unique_prompts_rollout)
        dataset_index, prompt_metadata = iter_dataloader.next_for_step(prefill_target_step)
        add_prompt_to_generator(
            train_dataset[dataset_index],
            dataset_index,
            iter_dataloader.epoch_number,
            resume_training_step,
            pending_queries_map,
            param_prompt_Q,
            generation_configs["train"],
            is_eval=False,
            prompt_metadata=prompt_metadata,
        )
    if checkpoint_state and "num_total_tokens" in checkpoint_state:
        num_total_tokens = checkpoint_state["num_total_tokens"]
        logger.info(f"Restored num_total_tokens: {num_total_tokens}")
    else:
        num_total_tokens = 0

    if args.eval_on_step_0 and args.local_eval_every >= 0 and eval_dataset is not None and resume_training_step <= 1:
        logger.info("[Main Thread] Triggering evaluation before first training step (eval_on_step_0=True)")
        for eval_index, eval_example in enumerate(eval_dataset):
            add_prompt_to_generator(
                eval_example,
                eval_index,
                iter_dataloader.epoch_number,
                0,
                eval_pending_queries_map,
                param_prompt_Q,
                generation_configs["eval"],
                is_eval=True,
            )
        maybe_evaluate(
            args,
            training_step=0,
            evaluation_inference_results_Q=evaluation_inference_results_Q,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            episode=episode,
            eval_pending_queries_map=eval_pending_queries_map,
            eval_generation_config=generation_configs["eval"],
            generate_metrics_Q=generate_metrics_Q,
            num_eval_prompts=len(eval_dataset),
            model_dims=model_dims,
            actor_manager=actor_manager,
        )

    # Benchmark evaluation at step 0
    benchmark_eval_every = args.benchmark_eval_every if args.benchmark_eval_every >= 0 else args.local_eval_every
    if (
        args.eval_on_step_0
        and benchmark_eval_every > 0
        and benchmark_dataset is not None
        and resume_training_step <= 1
    ):
        logger.info("[Main Thread] Triggering benchmark evaluation before first training step (eval_on_step_0=True)")
        for benchmark_index, benchmark_example in enumerate(benchmark_dataset):
            add_prompt_to_generator(
                benchmark_example,
                benchmark_index,
                iter_dataloader.epoch_number,
                0,
                benchmark_pending_queries_map,
                param_prompt_Q,
                generation_configs["benchmark"],
                is_eval=True,
                is_benchmark=True,
            )
        maybe_evaluate_benchmark(
            args,
            training_step=0,
            evaluation_inference_results_Q=evaluation_inference_results_Q,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            episode=episode,
            benchmark_pending_queries_map=benchmark_pending_queries_map,
            benchmark_generation_config=generation_configs["benchmark"],
            generate_metrics_Q=generate_metrics_Q,
            num_benchmark_prompts=len(benchmark_dataset),
            model_dims=model_dims,
            actor_manager=actor_manager,
        )

    training_start_time = time.perf_counter()  # Track overall training start time
    training_set_size = len(train_dataset)
    cumulative_filtered_prompts_zero = 0
    cumulative_total_prompts = 0
    for training_step in range(resume_training_step, args.num_training_steps + 1):
        start_time = time.perf_counter()

        if (
            training_step == resume_training_step
            or training_step % args.update_progress_every == 0
            or training_step == args.num_training_steps
        ):
            maybe_update_beaker_description(
                current_step=training_step,
                total_steps=args.num_training_steps,
                start_time=training_start_time,
                wandb_url=wandb_url,
            )

        # Check if any of the threads have raised an exception.
        health_check_start = time.perf_counter()
        health_check_fn()
        health_check_time = time.perf_counter() - health_check_start

        (
            collated_data,
            data_thread_metrics,
            num_total_tokens,
            num_step_tokens,
            prompt_lengths,
            response_lengths,
            num_filtered_prompts,
            prompt_pass_entries,
        ) = load_data_from_packing_thread(packed_sequences_Q, num_total_tokens, stop_event, health_check_fn)

        if (
            training_step % args.local_eval_every == 0
            and eval_dataset is not None
            and (args.eval_on_step_0 or training_step > 1)
        ):
            for eval_index, eval_example in enumerate(eval_dataset):
                add_prompt_to_generator(
                    eval_example,
                    eval_index,
                    iter_dataloader.epoch_number,
                    training_step,
                    eval_pending_queries_map,
                    param_prompt_Q,
                    generation_configs["eval"],
                    is_eval=True,
                )

        # Benchmark evaluation prompts
        benchmark_eval_freq = args.benchmark_eval_every if args.benchmark_eval_every >= 0 else args.local_eval_every
        if (
            benchmark_eval_freq > 0
            and training_step % benchmark_eval_freq == 0
            and benchmark_dataset is not None
            and (args.eval_on_step_0 or training_step > 1)
        ):
            for benchmark_index, benchmark_example in enumerate(benchmark_dataset):
                add_prompt_to_generator(
                    benchmark_example,
                    benchmark_index,
                    iter_dataloader.epoch_number,
                    training_step,
                    benchmark_pending_queries_map,
                    param_prompt_Q,
                    generation_configs["benchmark"],
                    is_eval=True,
                    is_benchmark=True,
                )

        if collated_data is None:
            continue

        episode += args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout

        for metrics_Q in [generate_metrics_Q, weight_sync_metrics_Q]:
            try:
                data_thread_metrics |= metrics_Q.get_nowait()
            except Empty:
                logger.info("[Main Thread] didn't get train generation metrics")

        data_thread_metrics["time/health_check"] = health_check_time

        batch_filtered_zero = int(data_thread_metrics.get("batch/filtered_prompts_zero", 0) or 0)
        batch_total_prompts = int(data_thread_metrics.get("batch/total_prompts", 0) or 0)

        cumulative_filtered_prompts_zero += batch_filtered_zero
        cumulative_total_prompts += batch_total_prompts
        cumulative_prompt_progress = cumulative_filtered_prompts_zero + cumulative_total_prompts
        prompt_progress_fraction = (
            float(cumulative_prompt_progress) / training_set_size if training_set_size > 0 else 0.0
        )

        data_thread_metrics["train/training_set_size"] = training_set_size
        data_thread_metrics["train/cumulative_filtered_prompts_zero"] = cumulative_filtered_prompts_zero
        data_thread_metrics["train/cumulative_total_prompts"] = cumulative_total_prompts
        data_thread_metrics["train/cumulative_prompt_progress"] = cumulative_prompt_progress
        data_thread_metrics["train/cumulative_prompt_progress_fraction"] = prompt_progress_fraction
        if iter_dataloader is not None:
            iter_dataloader.update_progress_fraction(prompt_progress_fraction)

        one_training_step(
            args,
            policy_group,
            collated_data,
            tokenizer,
            data_thread_metrics,
            episode,
            training_step,
            num_total_tokens,
            num_step_tokens,
            start_time,
            train_dataset,
            training_start_time,
            wandb_url,
            tc.chat_template_name,
            model_dims,
            prompt_lengths,
            response_lengths,
            actor_manager,
            iter_dataloader,
        )

        if prompt_pass_logger and prompt_pass_entries:
            prompt_pass_logger.update(prompt_pass_entries)
            prompt_pass_logger.save_and_log(wandb_step=episode)
        if prompt_reuse_logger and prompt_pass_entries:
            prompt_reuse_logger.update(prompt_pass_entries)
            prompt_reuse_logger.save()

        # Checkpoint after one_training_step (or even if it was skipped)
        # This ensures we checkpoint progress even if the exact checkpoint step has no data
        if (
            args.checkpoint_state_freq > 0
            and training_step % args.checkpoint_state_freq == 0
            and args.checkpoint_state_dir is not None
        ):
            with Timer("[Main Thread] 🗡️ Saving checkpoint state"):
                # Save comprehensive client state including ShufflingIterator state
                client_state = {
                    "training_step": training_step,
                    "episode": episode,
                    "num_total_tokens": num_total_tokens,
                }

                # Save ShufflingIterator state
                if iter_dataloader is not None:
                    client_state["shuffling_iterator_state"] = iter_dataloader.get_state()

                ray_get_with_progress(
                    [
                        policy_group.models[i].save_checkpoint_state.remote(args.checkpoint_state_dir, client_state)
                        for i in range(args.world_size)
                    ],
                    desc=f"Saving checkpoint state at step {training_step}",
                )
                logger.info(f"Saved checkpoint state at step {training_step} to {args.checkpoint_state_dir}")

        logger.debug(f"[Main Thread] Triggered weight sync for step {training_step}")
        weight_sync_trigger_event.set()

        maybe_evaluate(
            args,
            training_step,
            evaluation_inference_results_Q,
            tokenizer,
            reward_fn,
            episode,
            eval_pending_queries_map,
            generation_configs["eval"],
            generate_metrics_Q,
            len(eval_dataset) if eval_dataset else 0,
            model_dims,
            actor_manager,
        )

        # Benchmark evaluation (uses same queue as eval, run sequentially)
        maybe_evaluate_benchmark(
            args,
            training_step,
            evaluation_inference_results_Q,
            tokenizer,
            reward_fn,
            episode,
            benchmark_pending_queries_map,
            generation_configs["benchmark"],
            generate_metrics_Q,
            len(benchmark_dataset) if benchmark_dataset else 0,
            model_dims,
            actor_manager,
        )

    if resume_training_step > args.num_training_steps:
        raise ValueError(f"Training didn't run since {resume_training_step=} > {args.num_training_steps=}")

    save_final_model(args, policy_group, tokenizer, training_step, wandb_url, tc.chat_template_name)


def main(args: Args, tc: TokenizerConfig, model_config: ModelConfig):
    tokenizer = make_tokenizer(tc, model_config)
    args = setup_runtime_variables(args)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)

    maybe_configure_local_eval_subset(args)

    beaker_config, wandb_url = setup_experiment_tracking(args, tc, model_config)

    train_dataset, eval_dataset, benchmark_dataset = setup_datasets(args, tc, tokenizer)

    if len(train_dataset) < (needed := max(args.async_steps, 1) * args.num_unique_prompts_rollout):
        raise ValueError(
            f"Train dataset is too small! Is {len(train_dataset)} prompts, but {needed} are needed to have enough prompts for bsz and prefill. Try reducing async_steps or num_unique_prompts_rollout, or increasing the dataset size."
        )

    if args.cache_dataset_only:
        return

    pprint([args, model_config])

    # Initialize Ray before creating Ray objects
    # Exclude package manager files to prevent Ray from creating a new venv with different versions
    # This ensures workers use the same packages as the main process
    ray.init(
        dashboard_host="0.0.0.0",
        runtime_env={
            "excludes": [".git/", "pyproject.toml", "uv.lock", "*.toml", "requirements*.txt"],
            "env_vars": dict(os.environ),
        },
    )

    # Create Ray queues.
    # Since we now send/receive individual prompts, queue size should accommodate
    # all prompts from async_steps + 1 training steps
    queue_size = (args.async_steps + 1) * args.num_unique_prompts_rollout
    inference_results_Q = ray_queue.Queue(maxsize=queue_size)
    param_prompt_Q = ray_queue.Queue(maxsize=queue_size)
    # We don't care if we ever hit the max, so we let the queue be unbounded.
    evaluation_inference_results_Q = ray_queue.Queue()

    policy_group, vllm_engines, tool_objects, resume_training_step, episode, actor_manager = (
        create_model_and_optimizer(
            args,
            tc,
            model_config,
            beaker_config,
            wandb_url,
            tokenizer,
            inference_results_Q,
            param_prompt_Q,
            evaluation_inference_results_Q,
        )
    )

    # Get the model dimensions from one of the engines without loading weights
    model_dims = ray.get(vllm_engines[0].get_model_dims.remote())
    log_gpu_assignments(policy_group, vllm_engines, args.with_tracking)

    generation_configs = create_generation_configs(args)

    checkpoint_state = None
    if args.checkpoint_state_dir and os.path.exists(args.checkpoint_state_dir):
        # Try to load the checkpoint state from the first rank
        checkpoint_path = os.path.join(args.checkpoint_state_dir, "global_0", "state.pt")
        if os.path.exists(checkpoint_path):
            checkpoint_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            logger.info(f"Loaded checkpoint state from {checkpoint_path}")

            episode = checkpoint_state["episode"]
            logger.info(f"Restored episode count: {episode}")

    train_dataset_idxs = np.arange(len(train_dataset))
    iter_dataloader = ShufflingIterator(
        train_dataset_idxs,
        1,
        seed=args.seed,
        enable_prompt_pass_curriculum=args.enable_prompt_pass_curriculum,
        zero_pass_curriculum_fraction=args.zero_pass_curriculum_fraction,
        prompt_pass_curriculum_05sort=args.prompt_pass_curriculum_05sort,
        prompts_per_step=args.num_unique_prompts_rollout,
        enable_prompt_replay=args.enable_prompt_replay,
        prompt_replay_fraction=args.prompt_replay_fraction,
        prompt_replay_cooldown_steps=args.prompt_replay_cooldown_steps,
        prompt_replay_max_reuse_time=args.prompt_replay_max_reuse_time,
        prompt_replay_min_pass_rate=args.prompt_replay_min_pass_rate,
        prompt_replay_max_pass_rate=args.prompt_replay_max_pass_rate,
    )

    if checkpoint_state and "shuffling_iterator_state" in checkpoint_state:
        iter_dataloader.set_state(checkpoint_state["shuffling_iterator_state"])
        logger.info("Restored ShufflingIterator state from checkpoint")

    # Create additional queues (main queues already created above)
    packed_sequences_Q = Queue(maxsize=args.async_steps)
    pending_queries_map = PendingQueriesMap()
    eval_pending_queries_map = PendingQueriesMap()
    benchmark_pending_queries_map = PendingQueriesMap()
    generate_metrics_Q = Queue(maxsize=args.async_steps)
    weight_sync_metrics_Q = Queue(maxsize=args.async_steps)

    reward_fn = make_reward_fn(args)

    stop_event = threading.Event()
    executor = futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="grpo")

    try:
        episode = run_training(
            args,
            tokenizer,
            train_dataset,
            eval_dataset,
            benchmark_dataset,
            policy_group,
            vllm_engines,
            generation_configs,
            iter_dataloader,
            reward_fn,
            resume_training_step,
            episode,
            wandb_url,
            tc,
            stop_event,
            executor,
            inference_results_Q,
            param_prompt_Q,
            evaluation_inference_results_Q,
            packed_sequences_Q,
            pending_queries_map,
            eval_pending_queries_map,
            benchmark_pending_queries_map,
            generate_metrics_Q,
            weight_sync_metrics_Q,
            actor_manager,
            model_dims,
            checkpoint_state,
        )
    except Exception as e:
        if args.send_slack_alerts:
            utils.send_slack_alert(e)
        raise
    finally:
        cleanup_training_resources(
            stop_event, executor, [inference_results_Q, param_prompt_Q, evaluation_inference_results_Q], actor_manager
        )

    # Ai2 logic: we use /output to store the artifacts of the job, so we
    # make a copy of the model to `/output` in the end.
    if (
        args.try_auto_save_to_beaker
        and is_beaker_job()
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and args.output_dir.rstrip("/") != "/output"
        and os.path.isdir(args.output_dir)
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)
    logger.info("finished training")

    accelerator = Namespace()
    accelerator.is_main_process = True  # hack
    if args.push_to_hub:
        logger.info("Pushing model to hub")
        push_folder_to_hub(accelerator, args.output_dir, args.hf_repo_id, args.hf_repo_revision)

    # Check for runtime leaks before exiting
    logger.info("Checking for runtime leaks...")

    utils.check_runtime_leaks()


if __name__ == "__main__":
    utils.check_oe_eval_internal()

    parser = ArgumentParserPlus((Args, TokenizerConfig, ModelConfig))
    args, tokenizer_config, model_config = parser.parse_args_into_dataclasses()
    assert isinstance(args, Args)
    assert isinstance(tokenizer_config, TokenizerConfig)
    assert isinstance(model_config, ModelConfig)

    main(args, tokenizer_config, model_config)
