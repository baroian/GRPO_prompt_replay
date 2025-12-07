#!/usr/bin/env python3
"""
visualize_px_table.py

Create a Sankey-style diagram (Matplotlib) showing how prompts move between
p_x buckets across training epochs, emit a per-epoch bucket-percentage table,
and render a heatmap of those percentages. Defaults to using
/home/rberger/rlvr/open-instruct/UC/p_x_tables/p_x_table_1241.csv and writes
outputs alongside that CSV. No Plotly/Kaleido required.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, Rectangle


RUN_DIR = Path("/home/rberger/rlvr/open-instruct/UC/p_x_tables/")
DEFAULT_INPUT_PATH = Path("/home/rberger/rlvr/open-instruct/UC/p_x_tables/qwen_px0_filter_0.5_8096prompts_11853/p_x_table_1731.csv")
DEFAULT_OUTPUT_NAME = "p_x_sankey_epochs.png"
DEFAULT_TABLE_NAME = "p_x_bucket_percentages.txt"
DEFAULT_HEATMAP_NAME = "p_x_bucket_percentages_heatmap.png"
NODE_WIDTH = 0.25
CURVE_POINTS = 50
HEATMAP_HIGH_PERCENT_THRESHOLD = 40.0


def bucket_sort_key(value: str) -> Tuple[float, str]:
    try:
        return (float(value), value)
    except ValueError:
        return (float("inf"), value)


def format_bucket_value(value: float, decimals: int) -> str:
    return f"{float(value):.{decimals}f}"


def round_epoch_columns(df: pd.DataFrame, epoch_cols: Sequence[str], decimal_places: int) -> pd.DataFrame:
    rounded = df.copy()
    for col in epoch_cols:
        rounded[col] = pd.to_numeric(rounded[col], errors="coerce").round(decimal_places)
    return rounded


@dataclass
class Transition:
    epoch_src: int
    bucket_src: str
    epoch_dst: int
    bucket_dst: str
    count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Matplotlib Sankey diagram from a p_x per-prompt table."
    )
    parser.add_argument(
        "--csv",
        "-c",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input CSV path (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--folder",
        "-f",
        type=Path,
        help="Input folder containing prompt_pass_rates.csv file. If provided, overrides --csv.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        help="Output folder for generated files. If omitted, uses the same folder as input.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Where to write the chart. If omitted, defaults to a PNG beside the CSV.",
    )
    parser.add_argument(
        "--precision",
        "-p",
        type=int,
        default=3,
        help="Decimal places to keep when grouping bucket values. Default: 3",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Ignore transitions that occur fewer times than this. Default: 1",
    )
    parser.add_argument(
        "--title",
        help="Optional custom chart title. Defaults to a title derived from the CSV filename.",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        help="Destination for the bucket percentage table (TXT). "
        "Defaults to p_x_bucket_percentages.txt beside the CSV.",
    )
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        help="Destination for the percentage heatmap PNG. "
        "Defaults to p_x_bucket_percentages_heatmap.png beside the CSV.",
    )
    return parser.parse_args()


def load_p_x_table(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path}")

    df = pd.read_csv(path)
    if "prompt_id" not in df.columns:
        raise ValueError("CSV must contain a 'prompt_id' column.")

    epoch_cols = get_epoch_columns(df)
    if len(epoch_cols) < 2:
        raise ValueError("Need at least two epoch_* columns to build a Sankey diagram.")

    for col in epoch_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, epoch_cols


def get_epoch_columns(df: pd.DataFrame) -> List[str]:
    epoch_cols: List[str] = []
    for column in df.columns:
        if column.startswith("epoch_"):
            try:
                int(column.split("_", 1)[1])
            except ValueError:
                continue
            epoch_cols.append(column)
    epoch_cols.sort(key=lambda col: int(col.split("_", 1)[1]))
    return epoch_cols


def extract_transitions(
    rounded_df: pd.DataFrame,
    epoch_cols: Sequence[str],
    decimal_places: int,
    min_count: int,
) -> List[Transition]:
    transitions: List[Transition] = []
    for col_a, col_b in zip(epoch_cols, epoch_cols[1:]):
        pair = rounded_df[[col_a, col_b]].dropna()
        if pair.empty:
            continue

        counts = pair.groupby([col_a, col_b]).size().astype(int)
        epoch_a = int(col_a.split("_", 1)[1])
        epoch_b = int(col_b.split("_", 1)[1])

        for (val_a, val_b), count in counts.items():
            if count < min_count:
                continue
            bucket_a = format_bucket_value(val_a, decimal_places)
            bucket_b = format_bucket_value(val_b, decimal_places)
            transitions.append(Transition(epoch_a, bucket_a, epoch_b, bucket_b, int(count)))

    if not transitions:
        raise ValueError("No valid transitions were found after filtering.")

    return transitions


def compute_bucket_percentages(
    rounded_df: pd.DataFrame,
    epoch_cols: Sequence[str],
    decimal_places: int,
) -> Tuple[List[str], Dict[Tuple[str, int], float]]:
    percentages: Dict[Tuple[str, int], float] = {}
    bucket_set = set()

    for col in epoch_cols:
        series = rounded_df[col].dropna()
        epoch_num = int(col.split("_", 1)[1])

        if series.empty:
            continue

        total = float(series.size)
        counts = series.value_counts()
        for value, count in counts.items():
            bucket = format_bucket_value(value, decimal_places)
            percentages[(bucket, epoch_num)] = (count / total) * 100.0
            bucket_set.add(bucket)

    bucket_list = sorted(bucket_set, key=bucket_sort_key)
    return bucket_list, percentages


def format_percentage_table(
    bucket_labels: Sequence[str],
    epoch_numbers: Sequence[int],
    percentages: Dict[Tuple[str, int], float],
) -> str:
    if not bucket_labels:
        return "No bucket data available."

    header = ["p(x)"] + [f"epoch {epoch}" for epoch in epoch_numbers]
    lines = ["\t".join(header)]

    for bucket in bucket_labels:
        row = [bucket]
        for epoch in epoch_numbers:
            value = percentages.get((bucket, epoch), 0.0)
            row.append(f"{value:5.2f}%")
        lines.append("\t".join(row))

    return "\n".join(lines)


def save_percentage_table(table_text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_text + "\n", encoding="utf-8")


def build_percentage_heatmap(
    bucket_labels: Sequence[str],
    epoch_numbers: Sequence[int],
    percentages: Dict[Tuple[str, int], float],
    title: str,
) -> plt.Figure:
    if not bucket_labels or not epoch_numbers:
        raise ValueError("Need bucket and epoch data to build heatmap.")

    data = np.zeros((len(bucket_labels), len(epoch_numbers)), dtype=float)
    for row, bucket in enumerate(bucket_labels):
        for col, epoch in enumerate(epoch_numbers):
            data[row, col] = percentages.get((bucket, epoch), 0.0)

    fig_width = max(6.5, len(epoch_numbers) * 0.9)
    fig_height = max(4.5, len(bucket_labels) * 0.45 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    mesh = ax.imshow(data, aspect="auto", cmap="viridis", origin="lower")
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.034, pad=0.04)
    cbar.set_label("% of prompts", rotation=90)

    ax.set_xticks(range(len(epoch_numbers)))
    ax.set_xticklabels([f"epoch {epoch}" for epoch in epoch_numbers], rotation=45, ha="right")
    ax.set_yticks(range(len(bucket_labels)))
    ax.set_yticklabels(bucket_labels)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("p(x) bucket")
    ax.set_title(title, fontsize=13, pad=16)

    for row in range(len(bucket_labels)):
        for col in range(len(epoch_numbers)):
            value = data[row, col]
            text_color = "white" if value < HEATMAP_HIGH_PERCENT_THRESHOLD else "black"
            ax.text(
                col,
                row,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

    fig.tight_layout()
    return fig


def build_sankey_figure(
    transitions: Sequence[Transition],
    epoch_numbers: Sequence[int],
    title: str,
) -> plt.Figure:
    active_epochs = {
        epoch for t in transitions for epoch in (t.epoch_src, t.epoch_dst)
    }
    ordered_epochs = [epoch for epoch in epoch_numbers if epoch in active_epochs]
    if len(ordered_epochs) < 2:
        raise ValueError("Need at least two epochs with valid transitions.")

    buckets_by_epoch: Dict[int, List[str]] = defaultdict(list)
    outgoing = defaultdict(float)
    incoming = defaultdict(float)

    for t in transitions:
        buckets_by_epoch[t.epoch_src].append(t.bucket_src)
        buckets_by_epoch[t.epoch_dst].append(t.bucket_dst)
        outgoing[(t.epoch_src, t.bucket_src)] += t.count
        incoming[(t.epoch_dst, t.bucket_dst)] += t.count

    for epoch, buckets in buckets_by_epoch.items():
        buckets_by_epoch[epoch] = sorted(set(buckets), key=bucket_sort_key)

    node_sizes: Dict[Tuple[int, str], float] = {}
    epoch_totals: Dict[int, float] = {}
    for epoch in ordered_epochs:
        total = 0.0
        for bucket in buckets_by_epoch[epoch]:
            size = max(outgoing[(epoch, bucket)], incoming[(epoch, bucket)])
            node_sizes[(epoch, bucket)] = size
            total += size
        epoch_totals[epoch] = total

    max_total = max(epoch_totals.values()) if epoch_totals else 0.0
    if max_total == 0:
        raise ValueError("Could not compute node sizes for Sankey layout.")

    bucket_positions: Dict[Tuple[int, str], Tuple[float, float]] = {}
    for epoch in ordered_epochs:
        y_offset = 0.0
        for bucket in buckets_by_epoch[epoch]:
            size = node_sizes[(epoch, bucket)]
            bucket_positions[(epoch, bucket)] = (y_offset, y_offset + size)
            y_offset += size

    epoch_positions = {epoch: idx for idx, epoch in enumerate(ordered_epochs)}

    unique_buckets = sorted(
        {bucket for buckets in buckets_by_epoch.values() for bucket in buckets},
        key=bucket_sort_key,
    )
    cmap = plt.get_cmap("tab20")
    colors = {
        bucket: cmap(idx % cmap.N) for idx, bucket in enumerate(unique_buckets)
    }

    fig_width = max(8.0, len(ordered_epochs) * 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))
    ax.set_xlim(-0.5, len(ordered_epochs) - 0.5)
    ax.set_ylim(0, max_total * 1.08)
    ax.axis("off")

    for epoch, x_center in epoch_positions.items():
        ax.text(
            x_center,
            max_total * 1.02,
            f"epoch {epoch}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
        for bucket in buckets_by_epoch[epoch]:
            y0, y1 = bucket_positions[(epoch, bucket)]
            height = y1 - y0
            rect = Rectangle(
                (x_center - NODE_WIDTH / 2, y0),
                NODE_WIDTH,
                height,
                facecolor=colors[bucket],
                edgecolor="#333333",
                linewidth=0.5,
                alpha=0.9,
            )
            ax.add_patch(rect)
            ax.text(
                x_center,
                y0 + height / 2,
                bucket,
                ha="center",
                va="center",
                fontsize=7,
                color="white",
            )

    used_outgoing = defaultdict(float)
    used_incoming = defaultdict(float)

    for epoch_a, epoch_b in zip(ordered_epochs, ordered_epochs[1:]):
        flows = [
            t
            for t in transitions
            if t.epoch_src == epoch_a and t.epoch_dst == epoch_b
        ]
        flows.sort(key=lambda t: (bucket_sort_key(t.bucket_src), bucket_sort_key(t.bucket_dst)))
        for t in flows:
            src_pos = bucket_positions[(t.epoch_src, t.bucket_src)]
            tgt_pos = bucket_positions[(t.epoch_dst, t.bucket_dst)]

            src_y0 = src_pos[0] + used_outgoing[(t.epoch_src, t.bucket_src)]
            src_y1 = src_y0 + t.count
            used_outgoing[(t.epoch_src, t.bucket_src)] += t.count

            tgt_y0 = tgt_pos[0] + used_incoming[(t.epoch_dst, t.bucket_dst)]
            tgt_y1 = tgt_y0 + t.count
            used_incoming[(t.epoch_dst, t.bucket_dst)] += t.count

            add_flow_patch(
                ax=ax,
                x0=epoch_positions[t.epoch_src],
                x1=epoch_positions[t.epoch_dst],
                y0_start=src_y0,
                y0_end=src_y1,
                y1_start=tgt_y0,
                y1_end=tgt_y1,
                color=colors[t.bucket_src],
            )

    ax.set_title(title, fontsize=13, pad=20)
    return fig


def add_flow_patch(
    ax,
    x0: float,
    x1: float,
    y0_start: float,
    y0_end: float,
    y1_start: float,
    y1_end: float,
    color,
    alpha: float = 0.35,
) -> None:
    xs = np.linspace(x0 + NODE_WIDTH / 2, x1 - NODE_WIDTH / 2, CURVE_POINTS)
    top = np.linspace(y0_end, y1_end, CURVE_POINTS)
    bottom = np.linspace(y0_start, y1_start, CURVE_POINTS)
    verts = np.vstack(
        [
            np.column_stack([xs, top]),
            np.column_stack([xs[::-1], bottom[::-1]]),
        ]
    )
    patch = Polygon(verts, closed=True, facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_patch(patch)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.folder:
        folder_path = args.folder.expanduser()
        if not folder_path.is_absolute():
            folder_path = (Path.cwd() / folder_path).resolve()
        csv_path = folder_path / "prompt_pass_rates.csv"
    else:
        csv_path = args.csv.expanduser()
        if not csv_path.is_absolute():
            csv_path = (Path.cwd() / csv_path).resolve()

    # Determine output directory
    if args.output_folder:
        output_dir = args.output_folder.expanduser()
        if not output_dir.is_absolute():
            output_dir = (Path.cwd() / output_dir).resolve()
    else:
        output_dir = csv_path.parent

    if args.output is None:
        output_path = output_dir / DEFAULT_OUTPUT_NAME
    else:
        output_path = args.output.expanduser()
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()

    if args.table_output is None:
        table_output = output_dir / DEFAULT_TABLE_NAME
    else:
        table_output = args.table_output.expanduser()
        if not table_output.is_absolute():
            table_output = (Path.cwd() / table_output).resolve()

    if args.heatmap_output is None:
        heatmap_output = output_dir / DEFAULT_HEATMAP_NAME
    else:
        heatmap_output = args.heatmap_output.expanduser()
        if not heatmap_output.is_absolute():
            heatmap_output = (Path.cwd() / heatmap_output).resolve()

    decimals = max(args.precision, 0)
    min_count = max(args.min_count, 1)

    df, epoch_cols = load_p_x_table(csv_path)
    rounded_df = round_epoch_columns(df, epoch_cols, decimals)
    transitions = extract_transitions(rounded_df, epoch_cols, decimals, min_count)
    bucket_labels, percentages = compute_bucket_percentages(rounded_df, epoch_cols, decimals)

    epoch_numbers = [int(col.split("_", 1)[1]) for col in epoch_cols]
    title = args.title or f"Flow of p_x per prompt across epochs ({csv_path.name})"
    fig = build_sankey_figure(transitions, epoch_numbers, title)

    save_figure(fig, output_path)
    print(f"Sankey diagram saved to: {output_path}")

    table_text = format_percentage_table(bucket_labels, epoch_numbers, percentages)
    save_percentage_table(table_text, table_output)
    print(f"Bucket percentage table saved to: {table_output}")

    if bucket_labels:
        heatmap_title = f"p_x bucket percentages ({csv_path.name})"
        heatmap_fig = build_percentage_heatmap(bucket_labels, epoch_numbers, percentages, heatmap_title)
        save_figure(heatmap_fig, heatmap_output)
        print(f"Bucket percentage heatmap saved to: {heatmap_output}")
    else:
        print("No bucket data available; skipping heatmap.")


if __name__ == "__main__":
    main()
