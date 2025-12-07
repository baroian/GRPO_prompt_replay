#!/usr/bin/env python3
"""
analyze_reuse_stats.py

Generate statistics for prompt reuse events from a prompt_reuse_events.csv file.
Computes reuse count statistics including distributions, min/max/avg values.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate statistics for prompt reuse events."
    )
    parser.add_argument(
        "--folder",
        "-f",
        type=Path,
        required=True,
        help="Input folder containing prompt_reuse_events.csv file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file for statistics. Defaults to reuse_stats.txt in the input folder.",
    )
    return parser.parse_args()


def load_reuse_events(path: Path) -> pd.DataFrame:
    """Load and validate the prompt_reuse_events.csv file."""
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path}")

    df = pd.read_csv(path)

    required_columns = ["prompt_id", "reuse_count"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV must contain a '{col}' column.")

    # Convert reuse_count to numeric, handling any NaN values
    df["reuse_count"] = pd.to_numeric(df["reuse_count"], errors="coerce").fillna(0).astype(int)

    return df


def compute_reuse_statistics(df: pd.DataFrame) -> str:
    """Compute comprehensive reuse count statistics."""
    reuse_counts = df["reuse_count"]

    # Basic statistics
    total_prompts = len(df)
    max_reuse = reuse_counts.max()
    min_reuse = reuse_counts.min()
    avg_reuse_all = reuse_counts.mean()

    # Statistics for prompts with reuse_count > 0
    reused_prompts = df[df["reuse_count"] > 0]
    count_reused = len(reused_prompts)
    avg_reuse_nonzero = reused_prompts["reuse_count"].mean() if count_reused > 0 else 0

    # Check for additional columns (like cooldown/training step info)
    last_col = df.columns[-1] if len(df.columns) > 7 else None
    last_col_stats = ""
    if last_col and last_col != "reuse_count":
        last_col_values = pd.to_numeric(df[last_col], errors="coerce").dropna()
        if len(last_col_values) > 0:
            last_col_stats = f"""
ADDITIONAL COLUMN STATISTICS ({last_col.strip()}):
  Max value: {last_col_values.max()}
  Min value: {last_col_values.min():.1f}
  Average: {last_col_values.mean():.1f}
  Non-null values: {len(last_col_values)} / {total_prompts} ({len(last_col_values)/total_prompts*100:.1f}%)
"""

    # Distribution of reuse counts
    reuse_distribution = Counter(reuse_counts)
    sorted_counts = sorted(reuse_distribution.items())

    # Generate the report
    lines = []
    lines.append("=" * 60)
    lines.append("PROMPT REUSE STATISTICS")
    lines.append("=" * 60)
    lines.append("")

    lines.append("BASIC STATISTICS:")
    lines.append(f"  Total prompts: {total_prompts:,}")
    lines.append(f"  Maximum reuse count: {max_reuse}")
    lines.append(f"  Minimum reuse count: {min_reuse}")
    lines.append(f"  Average reuse count (all prompts): {avg_reuse_all:.2f}")
    lines.append("")

    lines.append("REUSED PROMPTS STATISTICS:")
    lines.append(f"  Prompts with reuse_count > 0: {count_reused:,} ({count_reused/total_prompts*100:.1f}%)")
    if count_reused > 0:
        lines.append(f"  Average reuse count (non-zero): {avg_reuse_nonzero:.2f}")
    else:
        lines.append("  Average reuse count (non-zero): N/A (no reused prompts)")
    lines.append("")

    # Add additional column statistics if available
    if last_col_stats:
        lines.append(last_col_stats.strip())
        lines.append("")

    lines.append("REUSE COUNT DISTRIBUTION:")
    lines.append("  Count | Frequency | Percentage")
    lines.append("  ------|-----------|-----------")

    for count_val, frequency in sorted_counts:
        percentage = (frequency / total_prompts) * 100
        lines.append(f"       {count_val:>3d} | {frequency:>9,d} | {percentage:>9.2f}%")

    lines.append("")
    lines.append("TOP REUSE COUNTS (most frequent):")
    # Show top 10 most common reuse counts
    most_common = reuse_distribution.most_common(10)
    for count_val, frequency in most_common:
        percentage = (frequency / total_prompts) * 100
        lines.append(f"       {count_val:>3d} | {frequency:>9,d} | {percentage:>9.2f}%")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def save_statistics(stats_text: str, output_path: Path) -> None:
    """Save the statistics to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(stats_text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    folder_path = args.folder.expanduser()
    if not folder_path.is_absolute():
        folder_path = (Path.cwd() / folder_path).resolve()

    csv_path = folder_path / "prompt_reuse_events.csv"

    # Determine output path
    if args.output is None:
        output_path = folder_path / "reuse_stats.txt"
    else:
        output_path = args.output.expanduser()
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()

    # Load and analyze data
    df = load_reuse_events(csv_path)
    stats_text = compute_reuse_statistics(df)

    # Save results
    save_statistics(stats_text, output_path)
    print(f"Reuse statistics saved to: {output_path}")

    # Print summary to console
    reuse_counts = df["reuse_count"]
    total_prompts = len(df)
    count_reused = len(df[df["reuse_count"] > 0])

    print("\nSUMMARY:")
    print(f"  Total prompts: {total_prompts:,}")
    print(f"  Reused prompts: {count_reused:,} ({count_reused/total_prompts*100:.1f}%)")
    print(".2f")
    print(f"  Max reuse count: {reuse_counts.max()}")
    print(f"  Min reuse count: {reuse_counts.min()}")


if __name__ == "__main__":
    main()
