# ENABLE_PROMPT_REPLAY: Detailed Explanation

## Overview

`ENABLE_PROMPT_REPLAY` is a feature in the GRPO (Group Relative Policy Optimization) training system that intelligently reuses high-performing prompts during training. Instead of always sampling new prompts from the dataset, the system can "replay" prompts that have shown good performance (measured by pass rates) to provide more training signal on challenging but solvable problems.

## Core Concept

The idea is simple: **if a prompt has a good pass rate (not too easy, not too hard), reuse it multiple times during training** to help the model learn more effectively. This is particularly useful for prompts that are:
- Challenging enough to provide learning signal (not already solved)
- Solvable enough that the model can make progress (not impossibly hard)
- In the "sweet spot" around 50% pass rate (optimal difficulty)

## Key Components

### 1. Configuration Parameters

When `ENABLE_PROMPT_REPLAY=True`, several parameters control the behavior:

```python
prompt_replay_fraction: float = 0.5
# Maximum fraction of each batch that can be filled with replayed prompts
# Example: If prompts_per_step=4 and fraction=0.5, up to 2 prompts can be replayed

prompt_replay_cooldown_steps: int = 5
# Number of training steps a prompt must wait before it becomes eligible for replay again
# Prevents immediate re-replay of the same prompt

prompt_replay_max_reuse_time: int = 5
# Maximum number of times a prompt can be replayed before it is retired
# Set to 0 or negative to disable the limit

prompt_replay_min_pass_rate: float = 0.24
# Minimum pass rate (inclusive) required for a prompt to enter replay
# Prompts below this rate are considered too hard

prompt_replay_max_pass_rate: float = 0.7
# Maximum pass rate (inclusive) allowed for a prompt to enter replay
# Prompts above this rate are considered too easy
```

### 2. Data Structures

#### PromptPassEntry
Tracks the performance of each prompt:
```python
@dataclass
class PromptPassEntry:
    prompt_id: str                    # Unique identifier
    dataset_name: str                 # Source dataset
    dataset_index: int                 # Index in dataset
    epoch_number: int | None          # When it was seen
    training_step: int                 # Training step when evaluated
    pass_rate: float                  # Fraction of completions that passed (0.0-1.0)
    was_reused: bool = False          # Whether this was a replayed prompt
    reuse_count: int | None = None    # How many times it's been reused
    cooldown_ready_step: int | None = None  # When it can be reused again
```

#### PromptReplayMetadata
Tracks replay information for prompts being used:
```python
@dataclass
class PromptReplayMetadata:
    was_reused: bool = False          # Whether this prompt is being replayed
    reuse_count: int | None = None    # Current reuse count
    scheduled_step: int | None = None # Training step it's scheduled for
    cooldown_ready_step: int | None = None  # When it can be reused next
```

#### StepReplayState
Tracks replay budget for each training step:
```python
@dataclass
class StepReplayState:
    replay_budget: int                # Max number of replays allowed this step
    issued_total: int = 0             # Total prompts issued for this step
    issued_replay: int = 0            # Number of replays issued
    issued_new: int = 0               # Number of new prompts issued
    reserved_indices: set[int] = field(default_factory=set)  # Indices already used
```

## How It Works: Step-by-Step

### Phase 1: Initial Training (Collecting Pass Rates)

1. **Normal Training**: Prompts are sampled from the dataset normally
2. **Evaluation**: After generating completions, each prompt is evaluated
3. **Pass Rate Calculation**: For each prompt, calculate `pass_rate = mean(scores) / max_possible_score`
   - Example: If a prompt generates 4 completions and 2 pass, `pass_rate = 0.5`
4. **Storage**: Pass rates are stored in `prompt_pass_rates[dataset_index] = pass_rate`

### Phase 2: Replay Eligibility

When a prompt completes evaluation:

1. **Check Pass Rate Window**: 
   ```python
   if min_pass_rate <= pass_rate <= max_pass_rate:
       # Prompt is eligible for replay
   ```
   - Example: With `min=0.2, max=0.7`, a prompt with `pass_rate=0.5` is eligible
   - A prompt with `pass_rate=0.1` is too hard (below minimum)
   - A prompt with `pass_rate=0.9` is too easy (above maximum)

2. **Check Reuse Limit**:
   ```python
   if current_reuse_count < max_reuse_time:
       # Prompt can still be replayed
   ```

3. **Add to Replay Heap**: If eligible, add to priority queue (min-heap)
   - Priority is based on `distance = abs(pass_rate - 0.5)`
   - Prompts closer to 50% pass rate are prioritized
   - Ties are broken by reuse count (fewer reuses = higher priority)

### Phase 3: Prompt Selection During Training

When requesting prompts for a training step:

1. **Calculate Replay Budget**:
   ```python
   replay_budget = floor(prompts_per_step * replay_fraction)
   # Example: prompts_per_step=4, fraction=0.5 → budget=2
   ```

2. **For Each Prompt Request**:
   - Check if replay budget is available and not exhausted
   - Try to pop a candidate from the replay heap
   - Validate candidate:
     - Pass rate still in window
     - Not currently active in replay
     - Cooldown period has passed
     - Reuse limit not exceeded
   - If valid candidate found: return it as replay
   - Otherwise: sample a new prompt from dataset

3. **Track State**: Update step state to track replay usage

### Phase 4: Cooldown Management

After a prompt is replayed:

1. **Mark as Active**: Add to `_active_replay_indices` set
2. **Update Last Step**: Record `prompt_last_step[dataset_index] = current_step`
3. **Calculate Cooldown**: `cooldown_ready_step = current_step + cooldown_steps`
4. **On Completion**: When evaluation completes, remove from active set
   - The prompt won't be eligible again until `cooldown_ready_step` is reached

## Detailed Example Walkthrough

### Setup

```python
# Configuration
prompts_per_step = 4              # 4 prompts per training step
replay_fraction = 0.5             # Up to 50% can be replays
cooldown_steps = 5                # Wait 5 steps between reuses
max_reuse_time = 3                # Can reuse up to 3 times
min_pass_rate = 0.2               # Minimum 20% pass rate
max_pass_rate = 0.7               # Maximum 70% pass rate

# Dataset has prompts with indices 0-99
```

### Training Step 1: Initial Evaluation

**Step 1**: Sample 4 new prompts: `[10, 23, 45, 67]`

**Step 2**: Generate completions and evaluate:
- Prompt 10: 2/4 pass → `pass_rate = 0.5` ✅ (eligible)
- Prompt 23: 0/4 pass → `pass_rate = 0.0` ❌ (too hard)
- Prompt 45: 3/4 pass → `pass_rate = 0.75` ❌ (too easy)
- Prompt 67: 1/4 pass → `pass_rate = 0.25` ✅ (eligible)

**Step 3**: Add eligible prompts to replay heap:
```
Replay Heap (priority queue):
  - Prompt 10: distance = |0.5 - 0.5| = 0.0, reuse_count = 0
  - Prompt 67: distance = |0.25 - 0.5| = 0.25, reuse_count = 0
```

### Training Step 2: First Replay Opportunity

**Step 2**: Request 4 prompts for step 2

**Replay Budget**: `floor(4 * 0.5) = 2` replays available

**Selection Process**:
1. Request prompt 1: Check heap → Prompt 10 available (cooldown OK, reuse OK)
   - Return Prompt 10 as **REPLAY** (reuse_count = 1)
   - Budget: 2/2 replays used
2. Request prompt 2: Check heap → Prompt 67 available
   - Return Prompt 67 as **REPLAY** (reuse_count = 1)
   - Budget: 2/2 replays exhausted
3. Request prompt 3: Budget exhausted, sample new → Prompt 34
4. Request prompt 4: Budget exhausted, sample new → Prompt 78

**Result**: `[10 (replay), 67 (replay), 34 (new), 78 (new)]`

**State**:
- `prompt_last_step[10] = 2`
- `prompt_last_step[67] = 2`
- `prompt_reuse_counts[10] = 1`
- `prompt_reuse_counts[67] = 1`
- `_active_replay_indices = {10, 67}`

### Training Step 3: Cooldown Active

**Step 3**: Request 4 prompts for step 3

**Replay Budget**: 2 replays available

**Selection Process**:
1. Request prompt 1: Check heap → Prompt 10
   - Cooldown check: `current_step (3) - last_step (2) = 1 < cooldown (5)` ❌
   - Prompt 10 still in cooldown, skip
2. Check Prompt 67: Same cooldown issue
3. No valid replays available, sample 4 new prompts: `[12, 56, 89, 91]`

**Result**: `[12 (new), 56 (new), 89 (new), 91 (new)]`

### Training Step 7: Cooldown Expired

**Step 7**: Request 4 prompts for step 7

**Replay Budget**: 2 replays available

**Selection Process**:
1. Request prompt 1: Check Prompt 10
   - Cooldown: `7 - 2 = 5 >= 5` ✅
   - Reuse count: `1 < 3` ✅
   - Return Prompt 10 as **REPLAY** (reuse_count = 2)
2. Request prompt 2: Check Prompt 67
   - Cooldown: `7 - 2 = 5 >= 5` ✅
   - Return Prompt 67 as **REPLAY** (reuse_count = 2)
3. Request prompts 3-4: Sample new prompts

**Result**: `[10 (replay), 67 (replay), 15 (new), 42 (new)]`

### Training Step 12: Max Reuse Reached

**Step 12**: Request 4 prompts for step 12

**Previous**: Prompt 10 has been reused 2 times already

**Selection Process**:
1. Request prompt 1: Check Prompt 10
   - Cooldown: `12 - 7 = 5 >= 5` ✅
   - Reuse count: `2 < 3` ✅
   - Return Prompt 10 as **REPLAY** (reuse_count = 3)
2. Request prompt 2: Check Prompt 10 again
   - Reuse count: `3 >= 3` ❌ (max reuse reached)
   - Prompt 10 is now retired from replay
   - Check Prompt 67 instead → return as replay
3. Request prompts 3-4: Sample new prompts

**Result**: `[10 (replay, final), 67 (replay), 28 (new), 73 (new)]`

**After Step 12**: Prompt 10 is removed from replay heap (max reuse reached)

## Priority Selection Algorithm

The replay heap uses a min-heap with this priority tuple:
```python
(distance, pass_rate, reuse_snapshot, dataset_index)
```

Where:
- `distance = abs(pass_rate - 0.5)` - Closer to 50% = higher priority
- `pass_rate` - Used for tie-breaking (higher rate wins)
- `reuse_snapshot` - Reuse count when added (for validation)
- `dataset_index` - The actual prompt index

**Example Priority Order**:
```
Pass Rate | Distance | Priority Order
----------|----------|---------------
0.5       | 0.0      | 1st (best)
0.4       | 0.1      | 2nd
0.6       | 0.1      | 3rd
0.3       | 0.2      | 4th
0.7       | 0.2      | 5th
0.2       | 0.3      | 6th
0.8       | 0.3      | 7th
```

## Integration with Training Loop

### 1. Prompt Generation Thread

```python
# In accumulate_inference_batches()
for each completed prompt:
    # Calculate pass rate
    pass_rate = mean(scores) / max_score
    
    # Create entry
    entry = PromptPassEntry(
        dataset_index=idx,
        pass_rate=pass_rate,
        was_reused=metadata.was_reused,
        training_step=current_step
    )
    
    # Replenish queue with next prompt
    replay_target_step = training_step + async_steps
    dataset_index, prompt_metadata = iter_dataloader.next_for_step(replay_target_step)
    # This is where replay selection happens!
```

### 2. Data Preparation Thread

```python
# After collecting results
if enable_prompt_replay and prompt_pass_entries:
    iter_dataloader.update_prompt_pass_entries(prompt_pass_entries)
    # This updates pass rates and adds eligible prompts to replay heap
```

### 3. Main Training Loop

```python
# Initial prefilling
for i in range(async_steps * num_prompts):
    target_step = resume_step + (i // num_prompts)
    dataset_index, metadata = iter_dataloader.next_for_step(target_step)
    # Add prompt to generation queue
```

## Benefits

1. **Efficient Learning**: Focus training on prompts in the "sweet spot" of difficulty
2. **Better Sample Efficiency**: Reuse valuable prompts instead of always sampling new ones
3. **Stable Learning**: Prompts around 50% pass rate provide optimal learning signal
4. **Prevents Overfitting**: Cooldown and max reuse limits prevent excessive repetition

## Configuration Examples

### Conservative Replay
```bash
ENABLE_PROMPT_REPLAY=True
PROMPT_REPLAY_FRACTION=0.2          # Only 20% replays
PROMPT_REPLAY_COOLDOWN_STEPS=10     # Long cooldown
PROMPT_REPLAY_MAX_REUSE_TIME=2      # Few reuses
PROMPT_REPLAY_MIN_PASS_RATE=0.3     # Narrow window
PROMPT_REPLAY_MAX_PASS_RATE=0.6
```

### Aggressive Replay
```bash
ENABLE_PROMPT_REPLAY=True
PROMPT_REPLAY_FRACTION=0.5          # 50% replays
PROMPT_REPLAY_COOLDOWN_STEPS=3      # Short cooldown
PROMPT_REPLAY_MAX_REUSE_TIME=10     # Many reuses
PROMPT_REPLAY_MIN_PASS_RATE=0.2     # Wide window
PROMPT_REPLAY_MAX_PASS_RATE=0.8
```

### Focused on 50% Pass Rate
```bash
ENABLE_PROMPT_REPLAY=True
PROMPT_REPLAY_FRACTION=0.4
PROMPT_REPLAY_COOLDOWN_STEPS=5
PROMPT_REPLAY_MAX_REUSE_TIME=5
PROMPT_REPLAY_MIN_PASS_RATE=0.4     # Tight around 50%
PROMPT_REPLAY_MAX_PASS_RATE=0.6
```

## Edge Cases and Special Behaviors

1. **Zero Pass Rate**: Prompts with 0% pass rate are never added to replay heap
2. **Pass Rate Changes**: If a prompt's pass rate changes significantly, it may be removed from heap
3. **Heap Rebuilding**: Heap is rebuilt when state is restored from checkpoint
4. **Active Replay Tracking**: Prompts currently being evaluated are excluded from replay selection
5. **Reserved Indices**: Each step reserves indices to prevent duplicate selection within the step

## Monitoring

The system tracks:
- `prompt_pass_rates.csv`: Historical pass rates per prompt per epoch
- `reuse_stats.txt`: Replay statistics (reuse counts, cooldown states)
- W&B logging: Replay metrics and pass rate distributions

## Real-World Example: Your Configuration

Based on your `run_test copy.slurm` file, here's how replay works with your settings:

```bash
ENABLE_PROMPT_REPLAY=True
PROMPT_REPLAY_FRACTION=0.3              # 30% of prompts can be replays
PROMPT_REPLAY_COOLDOWN_STEPS=10         # Wait 10 steps between reuses
PROMPT_REPLAY_MAX_REUSE_TIME=5          # Can reuse up to 5 times
PROMPT_REPLAY_MIN_PASS_RATE=0.2         # Minimum 20% pass rate
PROMPT_REPLAY_MAX_PASS_RATE=0.7         # Maximum 70% pass rate
```

### Scenario: Training with 4 prompts per step

**Step 1**: Initial evaluation
- Prompts sampled: `[100, 250, 500, 750]`
- After evaluation:
  - Prompt 100: `pass_rate = 0.15` ❌ (below 0.2 minimum)
  - Prompt 250: `pass_rate = 0.45` ✅ (eligible, distance = 0.05 from 0.5)
  - Prompt 500: `pass_rate = 0.55` ✅ (eligible, distance = 0.05 from 0.5)
  - Prompt 750: `pass_rate = 0.80` ❌ (above 0.7 maximum)

**Replay Heap After Step 1**:
```
Priority Queue (min-heap):
  - Prompt 250: (distance=0.05, rate=0.45, reuse=0, idx=250)
  - Prompt 500: (distance=0.05, rate=0.55, reuse=0, idx=500)
```

**Step 2**: Request 4 prompts
- Replay budget: `floor(4 * 0.3) = 1` replay available
- Selection:
  1. Prompt 1: Pop from heap → Prompt 250 (cooldown OK, reuse OK)
     - Return Prompt 250 as **REPLAY** (reuse_count = 1)
     - Budget exhausted
  2. Prompt 2-4: Sample new prompts: `[150, 300, 600]`

**Result**: `[250 (replay), 150 (new), 300 (new), 600 (new)]`

**Step 3-11**: Cooldown period
- Prompt 250 cannot be replayed (last used at step 2, need to wait until step 12)
- All prompts are new samples

**Step 12**: Cooldown expired
- Prompt 250 becomes eligible again (`12 - 2 = 10 >= 10`)
- Can be replayed again

**Step 22**: After 5 reuses
- Prompt 250 has been reused 5 times
- Removed from replay heap (max reuse reached)
- No longer eligible for replay

### Visual Timeline

```
Step 1:  [100❌, 250✅, 500✅, 750❌]  → Heap: [250, 500]
Step 2:  [250🔄, 150, 300, 600]       → 250 reused (count=1)
Step 3:  [150, 200, 400, 800]         → Cooldown active
...
Step 12: [250🔄, 180, 350, 650]       → 250 reused (count=2)
Step 13: [190, 210, 450, 700]         → Cooldown active
...
Step 22: [250🔄, 220, 380, 720]       → 250 reused (count=5, final)
Step 23: [230, 240, 390, 730]         → 250 retired from heap
```

## Summary

`ENABLE_PROMPT_REPLAY` is a sophisticated system that:
1. Tracks prompt performance (pass rates)
2. Identifies optimal prompts (20-70% pass rate, prioritizing ~50%)
3. Intelligently reuses them with cooldown and limits
4. Integrates seamlessly with async training pipeline
5. Provides fine-grained control over replay behavior

This allows the model to focus training on prompts that are challenging but solvable, leading to more efficient learning.

## Key Takeaways

1. **Pass Rate Window**: Only prompts with pass rates between `min` and `max` are eligible
2. **Priority**: Prompts closest to 50% pass rate are prioritized
3. **Cooldown**: Prevents immediate re-replay (waits `cooldown_steps`)
4. **Max Reuse**: Limits how many times a prompt can be reused (prevents overfitting)
5. **Fraction Control**: `replay_fraction` controls how much of each batch can be replays
6. **Dynamic**: Pass rates update as model improves, automatically adjusting replay eligibility

