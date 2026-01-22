# Adding New Benchmark Datasets: OlympiadBench and MinervaMAth

## Problem Statement

The user wants to add two new benchmarks for training and evaluation:
- `Hothan/OlympiadBench`
- `math-ai/minervamath`

Currently, the codebase fails when trying to add these datasets because they have different column formats than what the data pipeline expects.

---

## Current Expected Format

The codebase expects all datasets (training, eval, benchmark) to have this exact format:

```python
{
    "messages": [{"role": "user", "content": "problem text..."}],  # Chat-style messages
    "ground_truth": "answer",  # The expected answer (string or list)
    "dataset": "math"  # Which verifier to use
}
```

This format is processed by `rlvr_tokenize_v3` in `open_instruct/dataset_transformation.py` (line ~1364).

### Verifier Types Available

The `dataset` field maps to verifiers in `open_instruct/ground_truth_utils.py`:
- `"math"` - MathVerifier (extracts boxed answers, Minerva format, LaTeX)
- `"gsm8k"` - GSM8KVerifier (extracts last number)
- `"strict_math"` - StrictMathVerifier (Minerva format only)
- `"ifeval"` - IFEvalVerifier
- `"string_matcher"` - StringMatcherVerifier
- `"string_f1"` - F1Verifier
- `"puzzle"` - PuzzleMatcherVerifier
- `"code"` - CodeVerifier

For math benchmarks, use `"math"` as the verifier.

---

## Dataset Formats to Support

### 1. HuggingFaceH4/MATH-500 (currently used but broken)

```python
# Load: load_dataset('HuggingFaceH4/MATH-500', split='test')
# Columns: ['problem', 'solution', 'answer', 'subject', 'level', 'unique_id']
{
    "problem": "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates...",
    "answer": "\\left( 3, \\frac{\\pi}{2} \\right)",
    "solution": "We have that $r = \\sqrt{0^2 + 3^2} = 3...",
    "subject": "Precalculus",
    "level": 2,
    "unique_id": "test/precalculus/807.json"
}
```

### 2. math-ai/minervamath

```python
# Load: load_dataset('math-ai/minervamath', split='test')
# Columns: ['question', 'answer']
# Total: 272 examples
{
    "question": "Each of the two Magellan telescopes has a diameter of $6.5 \\mathrm{~m}$...",
    "answer": "1.6"
}
```

### 3. Hothan/OlympiadBench

```python
# Load: load_dataset('Hothan/OlympiadBench', 'OE_TO_maths_en_COMP', split='train')
# NOTE: Only has 'train' split, no 'test' split!
# NOTE: Requires config name (e.g., 'OE_TO_maths_en_COMP')
# Available configs: 
#   - OE_TO_maths_en_COMP (674 examples, text-only math, English, Competition)
#   - OE_MM_maths_en_COMP (multimodal - skip)
#   - Many others for physics, Chinese, etc.

# Columns: ['id', 'question', 'solution', 'final_answer', 'context', 'image_1'..., 
#           'modality', 'difficulty', 'is_multiple_answer', 'unit', 'answer_type', 
#           'error', 'question_type', 'subfield', 'subject', 'language']
{
    "id": 1606,
    "question": "Xenia and Sergey play the following game...",
    "solution": ["Sergey can determine Xenia's number in 2 but not fewer moves..."],
    "final_answer": ["2"],  # NOTE: This is a LIST
    "modality": "Text-only",
    "answer_type": "Numerical",
    "subject": "Math",
    ...
}
```

---

## Proposed Solution: Dataset Format Adapters

### Step 1: Add Format Conversion Functions

Add new functions in `open_instruct/dataset_transformation.py`:

```python
def convert_math500_format(
    row: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    problem_key: str = "problem",
    answer_key: str = "answer",
    verifier_name: str = "math",
):
    """Convert MATH-500 style datasets to the expected RLVR format."""
    # Create messages format
    row["messages"] = [{"role": "user", "content": row[problem_key]}]
    # Set ground truth
    row["ground_truth"] = row[answer_key]
    # Set verifier
    row["dataset"] = verifier_name
    return row


def convert_minervamath_format(
    row: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    question_key: str = "question",
    answer_key: str = "answer",
    verifier_name: str = "math",
):
    """Convert minervamath style datasets to the expected RLVR format."""
    row["messages"] = [{"role": "user", "content": row[question_key]}]
    row["ground_truth"] = row[answer_key]
    row["dataset"] = verifier_name
    return row


def convert_olympiadbench_format(
    row: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    question_key: str = "question",
    answer_key: str = "final_answer",
    verifier_name: str = "math",
):
    """Convert OlympiadBench style datasets to the expected RLVR format."""
    row["messages"] = [{"role": "user", "content": row[question_key]}]
    # final_answer is a list, take first element or join
    answer = row[answer_key]
    if isinstance(answer, list):
        answer = answer[0] if len(answer) == 1 else answer
    row["ground_truth"] = answer
    row["dataset"] = verifier_name
    return row
```

### Step 2: Register Transform Functions

Add to `TRANSFORM_FNS` dictionary in `dataset_transformation.py`:

```python
TRANSFORM_FNS = {
    # ... existing transforms ...
    "convert_math500_format": (convert_math500_format, "map"),
    "convert_minervamath_format": (convert_minervamath_format, "map"),
    "convert_olympiadbench_format": (convert_olympiadbench_format, "map"),
}
```

### Step 3: Update Benchmark Loading in grpo_fast.py

The benchmark datasets need to use different transform functions. Modify around line 2996-3010 in `grpo_fast.py`:

**Option A: Add new argument for benchmark transforms**

Add new args in the `Args` dataclass:
```python
dataset_transform_fn_benchmark: list[str] = field(
    default_factory=lambda: ["convert_math500_format", "rlvr_tokenize_v1", "rlvr_max_length_filter_v1"]
)
"""Transform functions for benchmark datasets (may differ from training)."""
```

Then use this in the benchmark loading section.

**Option B: Auto-detect format based on dataset columns**

Add logic to detect the dataset format and apply appropriate transforms.

### Step 4: Update Shell Script Configuration

In `scripts/train/olmo3/qwen_math_1_5.sh`, update the benchmark configuration:

```bash
# Benchmark evaluation: separate full benchmark (e.g., MATH-500)
# Format: "dataset_name num_samples" pairs
BENCHMARK_EVALS="${BENCHMARK_EVALS:-HuggingFaceH4/MATH-500 128 math-ai/minervamath 272}"
BENCHMARK_EVAL_SPLITS="${BENCHMARK_EVAL_SPLITS:-test test}"

# For OlympiadBench, you need to specify config:
# BENCHMARK_EVALS="${BENCHMARK_EVALS:-Hothan/OlympiadBench:OE_TO_maths_en_COMP 674}"
# BENCHMARK_EVAL_SPLITS="${BENCHMARK_EVAL_SPLITS:-train}"  # Only has train split!
```

---

## Alternative: Pre-convert Datasets

Instead of modifying the codebase, you could create pre-converted datasets:

```python
from datasets import load_dataset, Dataset

def convert_and_push(source_name, source_split, target_name, converter_fn):
    ds = load_dataset(source_name, split=source_split)
    converted = ds.map(converter_fn)
    converted = converted.select_columns(["messages", "ground_truth", "dataset"])
    converted.push_to_hub(target_name, split="test")

# Example for minervamath
def convert_minerva(row):
    return {
        "messages": [{"role": "user", "content": row["question"]}],
        "ground_truth": row["answer"],
        "dataset": "math"
    }

convert_and_push("math-ai/minervamath", "test", "your-username/minervamath-rlvr", convert_minerva)
```

Then use the converted dataset directly.

---

## Files to Modify

1. **`open_instruct/dataset_transformation.py`**
   - Add conversion functions
   - Register in `TRANSFORM_FNS`

2. **`open_instruct/grpo_fast.py`**
   - Add `dataset_transform_fn_benchmark` argument OR
   - Add auto-detection logic for benchmark dataset formats

3. **`scripts/train/olmo3/qwen_math_1_5.sh`**
   - Update `BENCHMARK_EVALS` with new datasets
   - Update `BENCHMARK_EVAL_SPLITS` accordingly

---

## Testing

After implementation, test with:

```bash
# Test dataset loading
python -c "
from datasets import load_dataset
from open_instruct.dataset_transformation import convert_minervamath_format

ds = load_dataset('math-ai/minervamath', split='test[:5]')
print('Before:', ds[0])
converted = ds.map(convert_minervamath_format, fn_kwargs={'tokenizer': None})
print('After:', converted[0])
"
```

---

## Key Code Locations

| File | Line | Purpose |
|------|------|---------|
| `dataset_transformation.py` | 1364-1399 | `rlvr_tokenize_v3` - expects `messages`, `ground_truth`, `dataset` |
| `dataset_transformation.py` | 1429-1441 | `TRANSFORM_FNS` - registry of transform functions |
| `grpo_fast.py` | 2996-3013 | Benchmark dataset loading |
| `grpo_fast.py` | 162-168 | Benchmark args definition |
| `ground_truth_utils.py` | 936-970 | `build_all_verifiers` - creates verifier instances |
| `ground_truth_utils.py` | 201-250 | `MathVerifier` - the verifier to use for math benchmarks |

---

## IMPORTANT: Current Limitation & Next Step

### Current Limitation

The current implementation only supports **one transform function for ALL benchmark datasets**. This means you **cannot mix benchmarks with different column formats** in a single run.

For example, this will NOT work:
```bash
# BROKEN: These datasets have different column formats!
BENCHMARK_EVALS="HuggingFaceH4/MATH-500 128 math-ai/minervamath 272"
BENCHMARK_TRANSFORM_FN="convert_math500_format rlvr_tokenize_v1 rlvr_max_length_filter_v1"
# ^ This transform expects 'problem' column, but minervamath has 'question' column
```

Currently, you can only evaluate one benchmark at a time, or multiple benchmarks that happen to share the same column format.

### Next Step: Auto-Detection of Dataset Format

To properly support multi-benchmark evaluation, the code needs to **auto-detect the format** based on column names and apply the appropriate converter automatically.

**Proposed implementation in `dataset_transformation.py`:**

```python
def auto_convert_benchmark_format(
    row: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    verifier_name: str = "math",
):
    """Auto-detect dataset format and convert to RLVR format."""
    # Detect format based on available columns
    if "problem" in row:  # MATH-500 style
        question = row["problem"]
        answer = row["answer"]
    elif "question" in row and "final_answer" in row:  # OlympiadBench style
        question = row["question"]
        answer = row["final_answer"]
        if isinstance(answer, list):
            answer = answer[0] if len(answer) == 1 else answer
    elif "question" in row:  # minervamath style
        question = row["question"]
        answer = row["answer"]
    else:
        raise ValueError(f"Unknown dataset format. Columns: {list(row.keys())}")
    
    row["messages"] = [{"role": "user", "content": question}]
    row["ground_truth"] = answer
    row["dataset"] = verifier_name
    return row
```

**Then usage would be simple:**
```bash
BENCHMARK_EVALS="HuggingFaceH4/MATH-500 128 math-ai/minervamath 272 Hothan/OlympiadBench:OE_TO_maths_en_COMP 674"
BENCHMARK_EVAL_SPLITS="test test train"
BENCHMARK_TRANSFORM_FN="auto_convert_benchmark_format rlvr_tokenize_v1 rlvr_max_length_filter_v1"
```

**Status:** ✅ IMPLEMENTED - `auto_convert_benchmark_format` added to `dataset_transformation.py` and registered in `TRANSFORM_FNS`.

**Usage Example:**
```bash
# Evaluate on MATH-500 + minervamath in a single run
BENCHMARK_EVALS="HuggingFaceH4/MATH-500 128 math-ai/minervamath 272" \
BENCHMARK_EVAL_SPLITS="test test" \
BENCHMARK_TRANSFORM_FN="auto_convert_benchmark_format rlvr_tokenize_v1 rlvr_max_length_filter_v1" \
sbatch run_grpo.slurm

# Evaluate on all three benchmarks
BENCHMARK_EVALS="HuggingFaceH4/MATH-500 128 math-ai/minervamath 272 Hothan/OlympiadBench:OE_TO_maths_en_COMP 674" \
BENCHMARK_EVAL_SPLITS="test test train" \
BENCHMARK_TRANSFORM_FN="auto_convert_benchmark_format rlvr_tokenize_v1 rlvr_max_length_filter_v1" \
sbatch run_grpo.slurm
```

---

## Notes

1. **OlympiadBench only has `train` split** - you must use `split='train'` not `split='test'`
2. **OlympiadBench requires config name** - e.g., `'OE_TO_maths_en_COMP'` for text-only English math
3. **`final_answer` is a list** in OlympiadBench - need to handle appropriately
4. The `MathVerifier` should work for all these math datasets as it tries multiple extraction methods (boxed, Minerva format, LaTeX)

---

## Failure Modes Encountered

### 1. `transform_fn` / `transform_fn_args` Length Mismatch

**Error:** `AssertionError: transform_fn and transform_fn_args must have the same length`

**Cause:** When using 3 benchmark transform functions (e.g., `convert_math500_format`, `rlvr_tokenize_v1`, `rlvr_max_length_filter_v1`), the code was passing only 2 args entries (built for the default 2 training transforms).

**Fix:** Build `benchmark_transform_fn_args` dynamically based on which functions are in `benchmark_transform_fn`:
```python
benchmark_transform_fn_args = []
for fn_name in benchmark_transform_fn:
    if fn_name == "rlvr_tokenize_v1":
        benchmark_transform_fn_args.append({"system_prompt_override": ...})
    elif fn_name == "rlvr_max_length_filter_v1":
        benchmark_transform_fn_args.append({"max_prompt_token_length": ...})
    else:
        benchmark_transform_fn_args.append({})  # Converter functions need no args
```
