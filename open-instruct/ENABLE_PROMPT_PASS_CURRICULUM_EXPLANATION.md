# ENABLE_PROMPT_PASS_CURRICULUM: Detailed Explanation

## Overview

`ENABLE_PROMPT_PASS_CURRICULUM` is a curriculum learning feature in the GRPO (Group Relative Policy Optimization) training system that dynamically reorders prompts each epoch based on their historical pass rates. Instead of randomly sampling prompts, the system organizes them by difficulty to optimize learning efficiency.

## Core Concept

The fundamental idea is **curriculum learning**: start with easier prompts and gradually introduce harder ones, or focus on prompts at the optimal difficulty level. The system tracks how well the model performs on each prompt (measured by pass rate) and uses this information to reorder prompts for the next epoch.

**Key Insight**: Prompts with higher pass rates are generally "easier" for the model, while prompts with lower pass rates are "harder". By ordering prompts strategically, we can:
- Start with easier prompts to build confidence
- Focus on prompts at optimal difficulty (around 50% pass rate)
- Gradually reintroduce failed prompts (0% pass rate) to give the model another chance

## Key Components

### 1. Configuration Parameters

When `ENABLE_PROMPT_PASS_CURRICULUM=True`, several parameters control the behavior:

```python
enable_prompt_pass_curriculum: bool = False
# Master switch to enable/disable curriculum learning

zero_pass_curriculum_fraction: float = 0.25
# Fraction of zero-pass prompts from the previous epoch to schedule in the next epoch
# Example: If 100 prompts had 0% pass rate in epoch 1, 25 of them will appear in epoch 2
# This gives failed prompts another chance to be learned

prompt_pass_curriculum_05sort: bool = False
# If True: Prioritize prompts closest to 50% pass rate (optimal difficulty)
# If False: Prioritize prompts with highest pass rates first (easier prompts first)
```

### 2. Data Structures

#### Prompt Pass Rates Dictionary
```python
self.prompt_pass_rates: dict[int, float] = {}
# Maps dataset_index -> pass_rate (0.0 to 1.0)
# Example: {42: 0.75, 1337: 0.25, 999: 0.0}
#   - Prompt 42: 75% pass rate (easy)
#   - Prompt 1337: 25% pass rate (hard)
#   - Prompt 999: 0% pass rate (failed)
```

#### Zero-Pass Tracking
```python
self._zero_pass_heap: list[tuple[int, int, int]] = []
# Min-heap of (epoch_number, insertion_order, dataset_index)
# Tracks prompts that had 0% pass rate, ordered by when they failed
# Used to select which zero-pass prompts to reintroduce

self._zero_pass_lookup: dict[int, tuple[int, int]] = {}
# Maps dataset_index -> (epoch_number, insertion_order)
# Quick lookup to validate heap entries

self._zero_pass_epochs: defaultdict[int, set[int]] = defaultdict(set)
# Maps epoch_number -> set of dataset_indices with 0% pass rate
# Tracks which prompts failed in which epoch
```

#### Epoch Tracking
```python
self.prompt_last_epoch: dict[int, int] = {}
# Maps dataset_index -> last epoch number when prompt was seen
# Used for tracking and debugging

self.data: np.ndarray
# The current epoch's ordered list of dataset indices
# This is what gets iterated over during training
```

## How It Works: Step-by-Step

### Phase 1: Initial Training (Collecting Pass Rates)

1. **Normal Training**: Prompts are sampled from the dataset (initially random or shuffled)
2. **Evaluation**: After generating completions, each prompt is evaluated
3. **Pass Rate Calculation**: For each prompt, calculate `pass_rate = mean(scores) / max_possible_score`
   - Example: If a prompt generates 4 completions with scores [1.0, 0.0, 1.0, 0.0], `pass_rate = 0.5`
   - Example: If all completions fail, `pass_rate = 0.0`
   - Example: If all completions pass, `pass_rate = 1.0`
4. **Storage**: Pass rates are stored in `prompt_pass_rates[dataset_index] = pass_rate`

### Phase 2: Epoch Completion and Reordering

When an epoch completes, `_prepare_next_epoch()` is called:

#### Step 1: Collect Zero-Pass Prompts from Completed Epoch

```python
zero_count_set = self._zero_pass_epochs.pop(completed_epoch, set())
# Get all prompts that had 0% pass rate in the completed epoch
# Example: If epoch 1 had prompts [42, 1337, 999] with 0% pass rate:
#   zero_count_set = {42, 1337, 999}
```

#### Step 2: Calculate Zero-Pass Quota

```python
zero_quota = math.ceil(self.zero_pass_fraction * len(self._zero_pass_heap))
# Calculate how many zero-pass prompts to reintroduce based on TOTAL heap size
# This ensures fair selection across all accumulated failures, not just the most recent epoch
# Example: If zero_pass_fraction=0.5 and heap has 500 prompts (300 from epoch 1 + 200 from epoch 2):
#   zero_quota = ceil(0.5 * 500) = 250 prompts
#   This selects from ALL accumulated failures, prioritizing oldest first
```

#### Step 3: Select Zero-Pass Prompts

```python
selected_zero = self._select_zero_pass_prompts(zero_quota, active_set)
# Select prompts from the zero-pass heap (oldest failures first)
# This gives failed prompts another chance, but not all of them
```

**Example Selection Process**:
- Zero-pass heap contains: `[(epoch=0, order=1, idx=42), (epoch=0, order=2, idx=1337), ...]`
- Selects first 25 prompts from heap (oldest failures first)
- These prompts will appear at the END of the next epoch's order

#### Step 4: Categorize Remaining Prompts

```python
positives = []  # Prompts with pass_rate > 0.0
unknown = []    # Prompts with no pass rate recorded yet

for idx in active_indices:
    rate = self.prompt_pass_rates.get(idx)
    if rate is None:
        unknown.append(idx)  # Never seen before
    elif rate > 0.0:
        positives.append((rate, idx))  # Has positive pass rate
    # rate == 0.0 prompts are already in selected_zero
```

#### Step 5: Sort Prompts by Strategy

**Strategy A: Default Sort (Highest Pass Rate First)**
```python
if not self.curriculum_center_sort:
    positives.sort(key=lambda item: item[0], reverse=True)
    # Sort by pass rate descending
    # Example: [0.9, 0.75, 0.5, 0.25] -> prompts ordered easiest to hardest
```

**Strategy B: Center Sort (Closest to 50% First)**
```python
if self.curriculum_center_sort:
    positives.sort(key=lambda item: abs(item[0] - 0.5))
    # Sort by distance from 50% (closer = better)
    # Ties maintain their original order (stable sort)
    # Example: [0.5, 0.4, 0.6, 0.3, 0.7] -> prompts ordered by optimal difficulty
```

**Example Sorting**:

**Default Sort** (highest first):
```
Pass Rate | Order
----------|------
0.9       | 1st (easiest)
0.75      | 2nd
0.6       | 3rd
0.5       | 4th
0.4       | 5th
0.25      | 6th
0.1       | 7th
```

**Center Sort** (closest to 50% first):
```
Pass Rate | Distance from 0.5 | Order
----------|-------------------|------
0.5       | 0.0               | 1st (optimal)
0.45      | 0.05              | 2nd
0.55      | 0.05              | 3rd
0.4       | 0.1               | 4th
0.6       | 0.1               | 5th
0.3       | 0.2               | 6th
0.7       | 0.2               | 7th
```

#### Step 6: Combine and Set Epoch Order

```python
ordered = [idx for _, idx in positives] + unknown + selected_zero
# Final order: positive pass rates + unknown + zero-pass prompts
# Example: [easy_prompts..., new_prompts..., failed_prompts...]
```

**Final Epoch Structure**:
```
[Positive Pass Rate Prompts (sorted)] + [Unknown Prompts (shuffled)] + [Zero-Pass Prompts (selected)]
```

### Phase 3: Training with Reordered Prompts

During the next epoch:
1. Prompts are sampled in the new order
2. Easy prompts (or optimal difficulty prompts) come first
3. Unknown prompts come in the middle
4. Failed prompts from previous epoch come last
5. As prompts are evaluated, their pass rates are updated
6. The cycle repeats for the next epoch

## Detailed Example: Complete Training Cycle

Let's walk through a concrete example with 10 prompts over 3 epochs.

### Initial State (Before Epoch 0)
```
Dataset: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Pass Rates: {} (empty, no data yet)
Order: Random shuffle, e.g., [3, 7, 1, 9, 0, 4, 6, 2, 8, 5]
```

### Epoch 0: Initial Training
Prompts are evaluated in random order. Results:
```
Prompt | Pass Rate | Status
-------|-----------|-------
0      | 0.75      | Easy
1      | 0.50      | Medium
2      | 0.25      | Hard
3      | 0.90      | Very Easy
4      | 0.00      | Failed
5      | 0.60      | Medium-Easy
6      | 0.00      | Failed
7      | 0.40      | Medium-Hard
8      | 0.00      | Failed
9      | 0.30      | Hard
```

Zero-pass prompts recorded:
```
_zero_pass_epochs[0] = {4, 6, 8}
_zero_pass_heap = [(0, 0, 4), (0, 1, 6), (0, 2, 8)]
```

### Epoch 1: Curriculum Reordering

**Configuration**: `zero_pass_fraction=0.25`, `curriculum_center_sort=False`

**Step 1**: Calculate zero-pass quota
```
zero_count_set = {4, 6, 8}  # 3 prompts from epoch 0 (for tracking)
zero_quota = ceil(0.25 * len(heap)) = ceil(0.25 * 3) = ceil(0.75) = 1 prompt
# Note: Quota is based on total heap size, not just zero_count_set
```

**Step 2**: Select zero-pass prompts
```
selected_zero = [4]  # Oldest failure (first in heap)
```

**Step 3**: Categorize prompts
```
positives = [
    (0.90, 3),  # Very Easy
    (0.75, 0),  # Easy
    (0.60, 5),  # Medium-Easy
    (0.50, 1),  # Medium
    (0.40, 7),  # Medium-Hard
    (0.30, 9),  # Hard
    (0.25, 2),  # Hard
]
unknown = []  # All prompts have been seen
```

**Step 4**: Sort positives (default: highest first)
```
positives.sort(reverse=True)
# Order: [3, 0, 5, 1, 7, 9, 2]
```

**Step 5**: Combine
```
ordered = [3, 0, 5, 1, 7, 9, 2, 4]
# Note: Prompts 6 and 8 are excluded (not selected for zero-pass quota)
```

**Epoch 1 Order**: `[3, 0, 5, 1, 7, 9, 2, 4]`
- Starts with easiest prompts (0.90, 0.75, 0.60)
- Progresses to harder prompts (0.25)
- Ends with one failed prompt (0.00) for retry

### Epoch 1: Training Results

After training, new pass rates:
```
Prompt | Old Rate | New Rate | Change
-------|----------|----------|-------
3      | 0.90     | 0.95     | Improved
0      | 0.75     | 0.80     | Improved
5      | 0.60     | 0.65     | Improved
1      | 0.50     | 0.55     | Improved
7      | 0.40     | 0.45     | Improved
9      | 0.30     | 0.35     | Improved
2      | 0.25     | 0.30     | Improved
4      | 0.00     | 0.20     | Learned! (was failed, now passing)
```

Zero-pass prompts from epoch 1:
```
_zero_pass_epochs[1] = {}  # No new failures!
```

### Epoch 2: Curriculum Reordering

**Step 1**: Zero-pass quota
```
zero_count_set = {}  # No failures in epoch 1
zero_quota = 0
selected_zero = []
```

**Step 2**: Categorize (all prompts have positive rates now)
```
positives = [
    (0.95, 3),
    (0.80, 0),
    (0.65, 5),
    (0.55, 1),
    (0.45, 7),
    (0.35, 9),
    (0.30, 2),
    (0.20, 4),  # Previously failed, now passing!
]
```

**Step 3**: Sort (highest first)
```
Order: [3, 0, 5, 1, 7, 9, 2, 4]
```

**Epoch 2 Order**: `[3, 0, 5, 1, 7, 9, 2, 4]`
- All prompts now have positive pass rates
- Still ordered easiest to hardest
- Model continues to improve on all prompts

### Example with Center Sort

If `curriculum_center_sort=True` for Epoch 1:

**Step 4**: Sort positives (center sort)
```python
positives.sort(key=lambda item: abs(item[0] - 0.5))
```

**Before sorting**: `[(0.90, 3), (0.75, 0), (0.60, 5), (0.50, 1), (0.40, 7), (0.30, 9), (0.25, 2)]`

**After sorting** (ties maintain original order):
```
Pass Rate | Distance from 0.5 | Order
----------|-------------------|------
0.50      | 0.00              | 1st (exactly 50%)
0.40      | 0.10              | 2nd
0.60      | 0.10              | 3rd
0.30      | 0.20              | 4th
0.75      | 0.25              | 5th
0.25      | 0.25              | 6th
0.90      | 0.40              | 7th
```

**Epoch 1 Order (Center Sort)**: `[1, 7, 5, 9, 0, 2, 3, 4]`
- Focuses on prompts at optimal difficulty (around 50%)
- These prompts provide the best learning signal

### Example: Multi-Epoch Accumulation (50% Fraction)

This example demonstrates how the quota calculation works across multiple epochs with accumulated failures:

**Epoch 1:**
- 600 prompts have 0% pass rate
- Added to heap: `_zero_pass_heap` contains 600 prompts from epoch 1
- `zero_count_set = {600 prompts}` (for tracking)

**Epoch 2 Preparation:**
- `zero_quota = ceil(0.5 * 600) = 300` (based on total heap size)
- Selects 300 prompts from heap (oldest first, i.e., epoch 1 failures)
- Selected prompts removed from heap
- **Remaining heap**: 300 prompts from epoch 1

**Epoch 2:**
- 200 new prompts have 0% pass rate
- Added to heap: `_zero_pass_heap` now contains 300 (old) + 200 (new) = **500 total**

**Epoch 3 Preparation:**
- `zero_count_set = {200 prompts}` (from epoch 2, for tracking only)
- `zero_quota = ceil(0.5 * 500) = 250` (based on **total heap size**, not just epoch 2)
- Selects 250 prompts from heap (oldest first)
- Since heap is min-heap ordered by `(epoch_number, insertion_order)`:
  - First 250 selected will be: 250 from epoch 1 (oldest) + 0 from epoch 2
  - **Remaining heap**: 50 from epoch 1 + 200 from epoch 2 = 250 total

**Key Benefit**: The quota is calculated from the total accumulated failures, ensuring fair selection across all epochs. Prompts from older epochs are prioritized (oldest first), but the quota accounts for all failures, preventing accumulation issues.

## Integration with Training Loop

### 1. Pass Rate Collection

```python
# In accumulate_inference_batches()
for each completed prompt:
    # Calculate pass rate
    pass_rate = mean(scores) / max_score
    
    # Create entry
    entry = PromptPassEntry(
        dataset_index=idx,
        pass_rate=pass_rate,
        training_step=current_step,
        epoch_number=current_epoch
    )
    
    prompt_pass_entries.append(entry)
```

### 2. Pass Rate Update

```python
# After collecting results
if args.enable_prompt_pass_curriculum and prompt_pass_entries:
    iter_dataloader.update_prompt_pass_entries(prompt_pass_entries)
    # This updates:
    #   - prompt_pass_rates[dataset_index] = pass_rate
    #   - prompt_last_epoch[dataset_index] = epoch_number
    #   - Records zero-pass prompts in heap
```

### 3. Epoch Transition

```python
# When epoch completes
self.epoch_number += 1
self._pending_epoch_completions.append(completed_epoch)
self._maybe_prepare_epochs()  # Triggers _prepare_next_epoch()
```

### 4. Prompt Sampling

```python
# During training, prompts are sampled in curriculum order
idx = self._raw_next_index()  # Returns next index from self.data
# self.data contains the reordered prompts for current epoch
```

## Benefits

1. **Efficient Learning**: Focus training on prompts at optimal difficulty
2. **Progressive Difficulty**: Start easy, gradually increase challenge
3. **Failed Prompt Recovery**: Give failed prompts another chance without overwhelming the model
4. **Adaptive**: Curriculum adjusts as model improves
5. **Flexible Strategies**: Choose between easy-first or optimal-difficulty-first

## Configuration Examples

### Conservative Curriculum (Easy First)
```bash
ENABLE_PROMPT_PASS_CURRICULUM=True
ZERO_PASS_CURRICULUM_FRACTION=0.1          # Only 10% of failed prompts retry
PROMPT_PASS_CURRICULUM_05SORT=False        # Easy prompts first
```

**Behavior**:
- Starts with highest pass rate prompts
- Gradually introduces harder prompts
- Minimal retry of failed prompts

### Aggressive Curriculum (Optimal Difficulty Focus)
```bash
ENABLE_PROMPT_PASS_CURRICULUM=True
ZERO_PASS_CURRICULUM_FRACTION=0.5          # 50% of failed prompts retry
PROMPT_PASS_CURRICULUM_05SORT=True          # Focus on 50% pass rate prompts
```

**Behavior**:
- Prioritizes prompts closest to 50% pass rate
- These provide optimal learning signal
- More aggressive retry of failed prompts

### Balanced Curriculum
```bash
ENABLE_PROMPT_PASS_CURRICULUM=True
ZERO_PASS_CURRICULUM_FRACTION=0.25          # 25% of failed prompts retry
PROMPT_PASS_CURRICULUM_05SORT=False         # Easy prompts first
```

**Behavior**:
- Default configuration
- Good balance between easy learning and challenge
- Moderate retry of failed prompts

## Edge Cases and Special Behaviors

### 1. No Pass Rate Data
- Prompts without pass rate data are placed in the "unknown" category
- Unknown prompts are shuffled randomly and placed after positive-rate prompts
- Once evaluated, they move to the appropriate category

### 2. All Prompts Failed (0% Pass Rate)
- If all prompts have 0% pass rate, `zero_quota` will be large
- System will select prompts from zero-pass heap (oldest failures first)
- Order will be: `selected_zero` (all failed prompts, oldest first)

### 3. Empty Dataset
- If no active indices remain, epoch data is set to empty array
- System falls back to random order with warning

### 4. Zero-Pass Fraction = 0
- No failed prompts are reintroduced
- Failed prompts are effectively excluded until they get a positive pass rate
- Useful for aggressive filtering of difficult prompts

### 5. Zero-Pass Fraction = 1.0
- All failed prompts are reintroduced
- Maximum retry opportunity for failed prompts
- May slow down training if many prompts consistently fail

## Comparison with ENABLE_PROMPT_REPLAY

| Feature | ENABLE_PROMPT_PASS_CURRICULUM | ENABLE_PROMPT_REPLAY |
|---------|------------------------------|---------------------|
| **Scope** | Epoch-level reordering | Step-level reuse |
| **Frequency** | Once per epoch | Multiple times per step |
| **Focus** | Ordering prompts by difficulty | Reusing high-value prompts |
| **Zero-Pass Handling** | Reintroduces fraction of failed prompts | Excludes zero-pass prompts |
| **Use Case** | Curriculum learning (easy→hard) | Sample efficiency (reuse good prompts) |

**Can Be Used Together**: These features complement each other:
- Curriculum reorders prompts each epoch
- Replay reuses prompts within steps
- Both use the same pass rate tracking system

## Monitoring

The system tracks:
- `prompt_pass_rates`: Historical pass rates per prompt
- `prompt_last_epoch`: Last epoch each prompt was seen
- Zero-pass heap: Failed prompts waiting for retry
- Epoch ordering: Current epoch's prompt order

## Real-World Example: Your Configuration

Based on your codebase, here's how curriculum works with typical settings:

```bash
ENABLE_PROMPT_PASS_CURRICULUM=True
ZERO_PASS_CURRICULUM_FRACTION=0.25          # 25% of failed prompts retry
PROMPT_PASS_CURRICULUM_05SORT=False         # Easy prompts first
```

**Training Flow**:
1. **Epoch 0**: Random order, collect initial pass rates
2. **Epoch 1**: Reorder by pass rate (highest first), reintroduce 25% of failed prompts at end
3. **Epoch 2**: Reorder again based on updated pass rates
4. **Continues**: Curriculum adapts as model improves

**Expected Behavior**:
- Model sees easier prompts first each epoch
- Failed prompts get another chance (25% of them)
- Ordering becomes more challenging as model improves
- System adapts to model's current capability level

## Code Flow Summary

```
Training Step
    ↓
Evaluate Prompts → Calculate Pass Rates
    ↓
Update prompt_pass_rates dictionary
    ↓
Epoch Completes
    ↓
_prepare_next_epoch(completed_epoch)
    ↓
1. Get zero-pass prompts from completed epoch
2. Calculate zero-pass quota (fraction * count)
3. Select zero-pass prompts from heap
4. Categorize: positives, unknown, zero-pass
5. Sort positives (by strategy)
6. Combine: positives + unknown + selected_zero
    ↓
Set self.data = ordered indices
    ↓
Next Epoch: Sample prompts in curriculum order
```

## Conclusion

`ENABLE_PROMPT_PASS_CURRICULUM` implements a sophisticated curriculum learning system that:
- Dynamically reorders prompts based on performance
- Provides flexible strategies (easy-first or optimal-difficulty-first)
- Intelligently handles failed prompts
- Adapts as the model improves
- Integrates seamlessly with the GRPO training loop

This creates a more efficient learning process by ensuring the model sees prompts in an order that maximizes learning signal and gradually increases difficulty.

