"""
plot_varbatch.py — Line plots of QPS (default) or estimated total time vs batch size.

QPS values (mean/median/min/max/std) are pre-computed by parse_varbatch.py and read
directly from the CSV. Default metric is median QPS.

Usage:
    cd results
    python plot_varbatch.py --sf 1 --system sgs-gpu05
    python plot_varbatch.py --sf 1 --system sgs-gpu05 --metric mean
    python plot_varbatch.py --sf 1 --system sgs-gpu05 --total_time --total_queries 10000 --unit s
    python plot_varbatch.py --sf 1 --system sgs-gpu05 --show_values --no_grid --log_y
"""

import argparse
import os
import math
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# ---------------------------------------------------------------------------
# Constants (aligned with plot_caliper.py)
# ---------------------------------------------------------------------------

CASE_RENAME_MAPPING = {
    "0: CPU-CPU-CPU":         "CPU",
    "1: GPU-CPU-CPU":         "GPU (Host D+I)",
    "1P: GPU-CPU(P)-CPU(P)":  "GPU (Pinned D+I)",
    "2: GPU-CPU-GPU":         "GPU (Host D)",
    "3: GPU-GPU-GPU":         "GPU",
}

CASE_ORDER = ["0: CPU-CPU-CPU", "1: GPU-CPU-CPU", "1P: GPU-CPU(P)-CPU(P)",
              "2: GPU-CPU-GPU", "3: GPU-GPU-GPU"]

CASE_COLORS = {
    "0: CPU-CPU-CPU":         "#2c3e50",   # Dark navy
    "1: GPU-CPU-CPU":         "#e67e22",   # Orange
    "1P: GPU-CPU(P)-CPU(P)":  "#d35400",   # Darker orange (paired with case 1)
    "2: GPU-CPU-GPU":         "#2ecc71",   # Green
    "3: GPU-GPU-GPU":         "#e74c3c",   # Red
}

CASE_MARKERS = {
    "0: CPU-CPU-CPU":         "o",
    "1: GPU-CPU-CPU":         "s",
    "1P: GPU-CPU(P)-CPU(P)":  "P",
    "2: GPU-CPU-GPU":         "^",
    "3: GPU-GPU-GPU":         "D",
}

QUERY_COLORS = {
    "ann_reviews":  "#3498db",
    "ann_images":   "#e67e22",
    "enn_reviews":  "#2ecc71",
    "enn_images":   "#9b59b6",
    # Image selectivity variants (pre = ENN with filter, post = ANN with filter)
    "pre_images_low_sel":           "#c0392b",   # Red
    "pre_images_high_sel":          "#1abc9c",   # Teal
    "post_images_low_sel":          "#e74c3c",   # Bright red
    "post_images_high_sel":         "#16a085",   # Dark teal
    "pre_images_hybrid_low_sel":    "#d35400",   # Dark orange
    "pre_images_hybrid_high_sel":   "#27ae60",   # Dark green
    "post_images_hybrid_low_sel":   "#f39c12",   # Amber
    "post_images_hybrid_high_sel":  "#2980b9",   # Strong blue
}

QUERY_MARKERS = {
    "ann_reviews":  "o",
    "ann_images":   "s",
    "enn_reviews":  "^",
    "enn_images":   "D",
    "pre_images_low_sel":           "v",
    "pre_images_high_sel":          "<",
    "post_images_low_sel":          ">",
    "post_images_high_sel":         "p",
    "pre_images_hybrid_low_sel":    "h",
    "pre_images_hybrid_high_sel":   "H",
    "post_images_hybrid_low_sel":   "*",
    "post_images_hybrid_high_sel":  "X",
}

INDEX_COLORS = {
    "Flat":                            "#1b9e77",   # Teal-green (Dark2)
    "IVF1024,Flat":                    "#7570b3",   # Muted purple
    "IVF1024(H),Flat":                 "#3b3680",   # Darker purple (ATS host view)
    "IVF1024,PQ8":                     "#d95f02",   # Burnt orange
    "IVF4096,Flat":                    "#66a61e",   # Olive green
    "IVF4096(H),Flat":                 "#3e6811",   # Darker olive (ATS host view)
    "HNSW32,Flat":                     "#a6761d",   # Brown
    "Cagra,64,32,NN_DESCENT":         "#e7298a",   # Magenta-pink
    "Cagra(C),64,32,NN_DESCENT":     "#9467bd",   # Purple — graph cache + GPU data (owning)
    "Cagra(C+H),64,32,NN_DESCENT":   "#666666",   # Dark gray
}

INDEX_MARKERS = {
    "Flat":                            "o",
    "IVF1024,Flat":                    "s",
    "IVF1024(H),Flat":                 "s",
    "IVF1024,PQ8":                     "^",
    "IVF4096,Flat":                    "v",
    "IVF4096(H),Flat":                 "v",
    "HNSW32,Flat":                     "D",
    "Cagra,64,32,NN_DESCENT":         "P",
    "Cagra(C),64,32,NN_DESCENT":     "*",
    "Cagra(C+H),64,32,NN_DESCENT":   "X",
}

# Paper-style display names for indexes
INDEX_DISPLAY_NAMES = {
    "Flat":                            "Exhaustive",
    "IVF1024,Flat":                    "IVF1024",
    "IVF1024(H),Flat":                 "IVF1024(H)",
    "IVF1024,PQ8":                     "IVF1024-PQ8",
    "IVF4096,Flat":                    "IVF4096",
    "IVF4096(H),Flat":                 "IVF4096(H)",
    "HNSW32,Flat":                     "HNSW32",
    "Cagra,64,32,NN_DESCENT":         "CAGRA",
    "Cagra(C),64,32,NN_DESCENT":     "CAGRA(C)",
    "Cagra(C+H),64,32,NN_DESCENT":   "CAGRA(C+H)",
}


def get_index_display_name(index_name):
    """Return paper-style display name for an index, falling back to raw name."""
    return INDEX_DISPLAY_NAMES.get(index_name, index_name)


# ---------------------------------------------------------------------------
# Summary plot constants: pre/post query matching
# ---------------------------------------------------------------------------

# Map table -> {variant_key: (pre_query, post_query)}
SUMMARY_QUERY_GROUPS = {
    "images": {
        "base":     ("pre_images",          "post_images"),
        "low":      ("pre_images_low_sel",  "post_images_low_sel"),
        "high":     ("pre_images_high_sel", "post_images_high_sel"),
    },
    "reviews": {
        "base":     ("pre_reviews",          "post_reviews"),
        "low":      ("pre_reviews_low_sel",  "post_reviews_low_sel"),
        "high":     ("pre_reviews_high_sel", "post_reviews_high_sel"),
    },
}

# Map non-hybrid query -> hybrid counterpart
SUMMARY_HYBRID_MAP = {
    "pre_images":           "pre_images_hybrid",
    "post_images":          "post_images_hybrid",
    "pre_reviews":          "pre_reviews_hybrid",
    "post_reviews":         "post_reviews_hybrid",
    "pre_images_low_sel":   "pre_images_hybrid_low_sel",
    "post_images_low_sel":  "post_images_hybrid_low_sel",
    "pre_images_high_sel":  "pre_images_hybrid_high_sel",
    "post_images_high_sel": "post_images_hybrid_high_sel",
    "pre_reviews_low_sel":  "pre_reviews_hybrid_low_sel",
    "post_reviews_low_sel": "post_reviews_hybrid_low_sel",
    "pre_reviews_high_sel": "pre_reviews_hybrid_high_sel",
    "post_reviews_high_sel":"post_reviews_hybrid_high_sel",
}

# (case_key, display_title) — hybrid uses _hybrid queries with special case mapping
SUMMARY_SUBPLOT_CASES = [
    ("0: CPU-CPU-CPU",  "CPU"),
    ("1: GPU-CPU-CPU",  "GPU (Host D+I)"),
    ("hybrid",          "Hybrid"),
    ("3: GPU-GPU-GPU",  "GPU"),
]

# Hybrid queries run under these case values in the CSV
SUMMARY_HYBRID_CASE = {
    "pre":  "1: GPU-CPU-CPU",   # pre_*_hybrid → case 1
    "post": "2: GPU-CPU-GPU",   # post_*_hybrid → case 2
}

# Style tokens — sized for LaTeX papers (single-col ≈ 3.5", double-col ≈ 7.0")
FIGSIZE_SINGLE    = (3.5, 2.6)   # single-column figure (plots A, B, E)
FIGSIZE_DOUBLE_W  = 7.0           # double-column figure width (plots C, D, F)
FIGSIZE_H_PER_ROW = 2.2           # height per row for multi-subplot figures

TITLE_FONTSIZE      = 8
LABEL_FONTSIZE      = 8
TICK_FONTSIZE       = 7
VALUE_FONTSIZE      = 7
LEGEND_FONTSIZE     = 7
DEFAULT_FONT_WEIGHT = 'bold'
EDGE_COLOR          = 'black'
ALPHA_BAND          = 0.18
DPI                 = 300

BATCH_SIZES = [1, 10, 100, 1000, 10000]

# Color palettes for operator breakdown plots (aligned with plot_caliper.py)
BREAKDOWN1_SEGMENTS = ["RelOperators_ms", "VectorSearch_ms", "DataMovement_ms", "IndexMovement_ms", "Other_ms"]
BREAKDOWN1_LABELS   = ["Rel. Operators",  "Vector Search",   "Data Movement",   "Index Movement",   "Other"]
BREAKDOWN1_COLORS   = ["#3498db",         "#e67e22",         "#2ecc71",         "#e74c3c",          "#9b59b6"]

BREAKDOWN2_SEGMENTS = [
    "VectorSearch_ms", "Filter_ms",  "Project_ms", "Join_ms",   "GroupBy_ms",
    "OrderBy_ms",      "Limit_ms",   "Take_ms",    "LocalBroadcast_ms",
    "Scatter_ms",      "Gather_ms",  "Distinct_ms","Other_ms",
]
BREAKDOWN2_LABELS   = [
    "VectorSearch",    "Filter",     "Project",    "Join",      "GroupBy",
    "OrderBy",         "Limit",      "Take",       "LocalBroadcast",
    "Scatter",         "Gather",     "Distinct",   "Other",
]
BREAKDOWN2_COLORS   = [
    "#e67e22", "#5dade2", "#f1c40f", "#ff69b4", "#34495e",
    "#c0392b", "#2ecc71", "#7f8c8d", "#1abc9c",
    "#bdc3c7", "#bdc3c7", "#a29bfe", "#9b59b6",
]

# ---------------------------------------------------------------------------
# Layout paddings (tweak these to change spacing around titles/labels)
# ---------------------------------------------------------------------------
TITLE_PAD   = 3    # pad for axis-level titles (ax.set_title)
XLABEL_PAD  = 6    # labelpad for x-axis labels
YLABEL_PAD  = 3    # labelpad for y-axis labels
TIGHT_LAYOUT_PAD  = 0.5
TIGHT_LAYOUT_RECT = [0, 0, 1, 0.93]  # rect top reserved for suptitle (plots D/F)
SUPTITLE_Y              = 0.98
SUPTITLE_Y_SUMMARY_GRID = 1.01
SUBPLOTS_ADJUST = dict(left=0.10, bottom=0.12, right=0.98, top=0.88,
                       hspace=0.5, wspace=0.35)

# Subplot fontsizes (same as globals for LaTeX consistency)
SUBPLOT_TITLE_FONTSIZE  = 8
SUBPLOT_LABEL_FONTSIZE  = 8
SUBPLOT_TICK_FONTSIZE   = 7
SUBPLOT_LEGEND_FONTSIZE = 7

# Set by main() based on --format argument
_EXT = ".jpeg"
_K_VALUE = "100"
SHOW_TITLE    = False   # set True via --title
SHOW_STD_BAND = True    # set False via --no-std-band
_PER_QUERY    = True    # set False via --no-per-query


# ---------------------------------------------------------------------------
# Core computation helpers
# ---------------------------------------------------------------------------

def scale_to_total(mean_ms, std_ms, batch_size, total_queries):
    """Estimated total time [ms] for total_queries queries processed in batches."""
    scale     = 1.0 if batch_size >= total_queries else total_queries / batch_size
    n_batches = math.ceil(total_queries / batch_size)
    return mean_ms * scale, std_ms * math.sqrt(n_batches)


def compute_series(rows_df, metric, total_queries, qps_mode, unit_s=False):
    """Return (xs, ys, es) arrays for one (query, case) combination.

    In QPS mode: reads pre-computed {metric}_qps and std_qps from the CSV.
    In total-time mode: reads {metric}_ms and std_ms, then scales.
    """
    xs, ys, es = [], [], []
    for _, row in rows_df.iterrows():
        bs = int(row["batch_size"])
        if qps_mode:
            y = float(row[f"{metric}_qps"])
            e = float(row["std_qps"])
        else:
            m = float(row[f"{metric}_ms"])
            s = float(row["std_ms"])
            y, e = scale_to_total(m, s, bs, total_queries)
            if unit_s:
                y /= 1000.0
                e /= 1000.0
        xs.append(bs); ys.append(y); es.append(e)
    return np.array(xs), np.array(ys), np.array(es)



def fmt_value(v, qps_mode, unit_s=False):
    if qps_mode:
        return f"{v:.0f}" if v < 1000 else f"{v/1000:.1f}k"
    else:
        if unit_s:
            return f"{v:.2f}" if v < 10 else f"{v:.1f}"
        return f"{v:.0f}" if v < 10000 else f"{v/1000:.0f}k"


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def make_safe_name(s):
    return s.replace(",", "_").replace(" ", "_").replace(":", "")


def _fill_std_band(ax, xs, ys, es, color):
    """Draw a std-deviation shaded band (guarded by SHOW_STD_BAND)."""
    if SHOW_STD_BAND:
        ax.fill_between(xs, np.maximum(0, ys - es), ys + es, color=color, alpha=ALPHA_BAND)


def set_xaxis_batch(ax):
    ax.set_xscale("log")
    ax.set_xticks(BATCH_SIZES)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(0.7, 15000)


def apply_grid(ax, show_grid):
    if show_grid:
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
    else:
        ax.grid(False)


def annotate_points(ax, xs, ys, log_y, qps_mode, unit_s=False):
    y_min, y_max = ax.get_ylim()
    for x, y in zip(xs, ys):
        if not np.isfinite(y) or y <= 0:
            continue
        y_pos = y * 1.18 if log_y else y + (y_max - y_min) * 0.025
        ax.text(x, y_pos, fmt_value(y, qps_mode, unit_s),
                ha='center', va='bottom',
                fontsize=VALUE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)


def annotate_points_minimal(ax, series_list, log_y, qps_mode, unit_s=False,
                            ann_min_gap=15):
    """Annotate values across series, suppressing labels that would physically overlap.

    At each x position, values are considered top-to-bottom (highest first). The top
    value is always annotated. Each subsequent value is annotated only if its label
    position is at least ann_min_gap pixels away from every already-placed label at
    that x.

    ann_min_gap: minimum vertical pixel distance between labels (default 15).
      - 0   → show all (like plain --show_values)
      - 15  → suppress labels closer than 15 px (default)
      - 50  → need substantial gap; aggressive suppression
    """
    from collections import defaultdict
    per_x = defaultdict(list)
    for xs, ys in series_list:
        for x, y in zip(xs, ys):
            if np.isfinite(y) and y > 0:
                per_x[x].append(y)

    y_min_ax, y_max_ax = ax.get_ylim()
    xs_to_ann, ys_to_ann = [], []

    for x in sorted(per_x.keys()):
        vals = sorted(per_x[x], reverse=True)   # highest first → always place top

        placed_px = []   # pixel y-coords of already-placed labels at this x
        for y in vals:
            # Label is drawn slightly above the data point (mirrors annotate_points logic)
            label_y = y * 1.18 if log_y else y + (y_max_ax - y_min_ax) * 0.025
            # Convert to display (pixel) coordinates
            _, py = ax.transData.transform((x, label_y))

            if not placed_px:
                # Top value: always annotate
                placed_px.append(py)
                xs_to_ann.append(x)
                ys_to_ann.append(y)
            else:
                min_dist = min(abs(py - pp) for pp in placed_px)
                if min_dist >= ann_min_gap:
                    placed_px.append(py)
                    xs_to_ann.append(x)
                    ys_to_ann.append(y)

    if xs_to_ann:
        annotate_points(ax, np.array(xs_to_ann), np.array(ys_to_ann),
                        log_y, qps_mode, unit_s)


def expand_ylim_for_labels(ax, all_ys, log_y):
    y_min, y_max = ax.get_ylim()
    data_max = max((v for v in all_ys if np.isfinite(v) and v > 0), default=y_max)
    ax.set_ylim(bottom=y_min, top=data_max * (3.0 if log_y else 1.25))



def make_ylabel(qps_mode, unit_s=False):
    if qps_mode:
        return "Throughput [QPS]"
    return "Estimated Total Time [s]" if unit_s else "Estimated Total Time [ms]"


def make_title_prefix(qps_mode, total_queries):
    return "Throughput" if qps_mode else f"Estimated Total Time ({total_queries} queries)"


def get_table(query):
    if "reviews" in query:
        return "reviews"
    if "images" in query:
        return "images"
    return "other"


def _finalize_ax(ax, log_y, show_values, show_grid, all_ys,
                 xlabel, ylabel, title, legend=True, unit_s=False, qps_mode=False,
                 show_title=None):
    set_xaxis_batch(ax)
    if log_y:
        ax.set_yscale("log")
    if show_values:
        expand_ylim_for_labels(ax, all_ys, log_y)
    apply_grid(ax, show_grid)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT,
                  labelpad=XLABEL_PAD)
    ax.set_ylabel(ylabel,  fontsize=LABEL_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT,
                  labelpad=YLABEL_PAD)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    _st = show_title if show_title is not None else SHOW_TITLE
    if title and _st:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT, pad=TITLE_PAD)
    if legend:
        ax.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.9)


# ---------------------------------------------------------------------------
# Plot A: per-query comparison (one fig per index×query, lines = cases)
# ---------------------------------------------------------------------------

def plot_per_query(df, index_name, sf, system, total_queries, metric,
                   out_base, log_y, qps_mode, show_grid, show_values, unit_s=False,
                   show_values_minimal=False, ann_min_gap=15):
    queries = sorted(df["query"].unique())
    for query in queries:
        qdf = df[df["query"] == query]
        cases_present = [c for c in CASE_ORDER if c in qdf["case"].values]

        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        all_ys = []

        for case in cases_present:
            cdf = qdf[qdf["case"] == case].sort_values("batch_size")
            if cdf.empty:
                continue
            xs, ys, es = compute_series(cdf, metric, total_queries, qps_mode, unit_s)
            all_ys.extend(ys.tolist())
            color  = CASE_COLORS[case]
            marker = CASE_MARKERS[case]

            ax.plot(xs, ys, color=color, marker=marker, label=CASE_RENAME_MAPPING[case],
                    linewidth=2, markersize=7, markeredgecolor=EDGE_COLOR, markeredgewidth=0.6)
            _fill_std_band(ax, xs, ys, es, color)

        _finalize_ax(ax, log_y, show_values, show_grid, all_ys,
                     xlabel="Batch Size",
                     ylabel=make_ylabel(qps_mode, unit_s),
                     title=f"{make_title_prefix(qps_mode, total_queries)}\n{query},  {index_name},  SF={sf},  {system}")

        if show_values:
            _ann_series = []
            for case in cases_present:
                cdf = qdf[qdf["case"] == case].sort_values("batch_size")
                xs, ys, _ = compute_series(cdf, metric, total_queries, qps_mode, unit_s)
                _ann_series.append((xs, ys))
            if show_values_minimal:
                annotate_points_minimal(ax, _ann_series, log_y, qps_mode, unit_s, ann_min_gap)
            else:
                for xs, ys in _ann_series:
                    annotate_points(ax, xs, ys, log_y, qps_mode, unit_s)

        safe_idx = make_safe_name(index_name)
        out_dir  = os.path.join(out_base, f"sf_{sf}", system, safe_idx, "per_query")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{system}_sf{sf}_k{_K_VALUE}_{safe_idx}_per_query_{query}{_EXT}")
        fig.tight_layout(pad=TIGHT_LAYOUT_PAD)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT A] {out_path}")


# ---------------------------------------------------------------------------
# Plot B: per-case summary (one fig per index×case×table, lines = queries)
# ---------------------------------------------------------------------------

def _plot_per_case_table(cdf, case, table, queries, index_name, sf, system,
                         total_queries, metric, out_base, log_y, qps_mode,
                         show_grid, show_values, unit_s=False, show_values_minimal=False, ann_min_gap=15):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    all_ys = []

    for query in queries:
        qdf = cdf[cdf["query"] == query].sort_values("batch_size")
        if qdf.empty:
            continue
        xs, ys, es = compute_series(qdf, metric, total_queries, qps_mode, unit_s)
        all_ys.extend(ys.tolist())
        color  = QUERY_COLORS.get(query)
        marker = QUERY_MARKERS.get(query, "o")

        ax.plot(xs, ys, color=color, marker=marker, label=query,
                linewidth=2, markersize=7, markeredgecolor=EDGE_COLOR, markeredgewidth=0.6)
        _fill_std_band(ax, xs, ys, es, color)

    case_label = CASE_RENAME_MAPPING[case]
    _finalize_ax(ax, log_y, show_values, show_grid, all_ys,
                 xlabel="Batch Size",
                 ylabel=make_ylabel(qps_mode, unit_s),
                 title=f"{make_title_prefix(qps_mode, total_queries)}\n{case_label},  {index_name},  {table},  SF={sf},  {system}")

    if show_values:
        _ann_series = []
        for query in queries:
            qdf = cdf[cdf["query"] == query].sort_values("batch_size")
            if qdf.empty:
                continue
            xs, ys, _ = compute_series(qdf, metric, total_queries, qps_mode, unit_s)
            _ann_series.append((xs, ys))
        if show_values_minimal:
            annotate_points_minimal(ax, _ann_series, log_y, qps_mode, unit_s, ann_min_gap)
        else:
            for xs, ys in _ann_series:
                annotate_points(ax, xs, ys, log_y, qps_mode, unit_s)

    safe_idx  = make_safe_name(index_name)
    safe_case = make_safe_name(case)
    out_dir = os.path.join(out_base, f"sf_{sf}", system, safe_idx, "per_case", table)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{system}_sf{sf}_k{_K_VALUE}_{safe_idx}_per_case_{table}_{safe_case}{_EXT}")
    fig.tight_layout(pad=TIGHT_LAYOUT_PAD)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT B] {out_path}")


def plot_per_case(df, index_name, sf, system, total_queries, metric,
                  out_base, log_y, qps_mode, show_grid, show_values, unit_s=False,
                  show_values_minimal=False, ann_min_gap=15):
    cases_present = [c for c in CASE_ORDER if c in df["case"].values]
    tables = {}
    for q in sorted(df["query"].unique()):
        tables.setdefault(get_table(q), []).append(q)

    for case in cases_present:
        cdf = df[df["case"] == case]
        for table, queries in sorted(tables.items()):
            _plot_per_case_table(cdf, case, table, queries, index_name, sf, system,
                                 total_queries, metric, out_base, log_y,
                                 qps_mode, show_grid, show_values, unit_s,
                                 show_values_minimal, ann_min_gap)


# ---------------------------------------------------------------------------
# Plot C: summary grid per table (rows=queries, cols=cases, per index×table)
# ---------------------------------------------------------------------------

def _plot_summary_grid_table(df, table, queries, cases, index_name, sf, system,
                              total_queries, metric, out_base, log_y, qps_mode,
                              show_grid, show_values, unit_s=False):
    n_rows = len(queries)
    n_cols = len(cases)
    if n_rows == 0 or n_cols == 0:
        return

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(FIGSIZE_DOUBLE_W, FIGSIZE_H_PER_ROW * n_rows),
                             squeeze=False)

    for r, query in enumerate(queries):
        for c, case in enumerate(cases):
            ax = axes[r][c]
            sub = df[(df["query"] == query) & (df["case"] == case)].sort_values("batch_size")

            if sub.empty:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center',
                        transform=ax.transAxes, fontsize=TICK_FONTSIZE, color='gray')
                ax.set_xticks([]); ax.set_yticks([])
            else:
                xs, ys, es = compute_series(sub, metric, total_queries, qps_mode, unit_s)
                color  = CASE_COLORS[case]
                marker = CASE_MARKERS[case]

                ax.plot(xs, ys, color=color, marker=marker,
                        linewidth=1.8, markersize=5,
                        markeredgecolor=EDGE_COLOR, markeredgewidth=0.5)
                _fill_std_band(ax, xs, ys, es, color)

                set_xaxis_batch(ax)
                if log_y:
                    ax.set_yscale("log")
                if show_values:
                    expand_ylim_for_labels(ax, ys.tolist(), log_y)
                apply_grid(ax, show_grid)
                ax.tick_params(axis='both', labelsize=SUBPLOT_TICK_FONTSIZE)

                if show_values:
                    annotate_points(ax, xs, ys, log_y, qps_mode, unit_s)

            if r == 0:
                ax.set_title(CASE_RENAME_MAPPING[case], fontsize=SUBPLOT_TITLE_FONTSIZE,
                             fontweight=DEFAULT_FONT_WEIGHT, pad=TITLE_PAD)
            if c == 0:
                ax.set_ylabel(query, fontsize=SUBPLOT_TITLE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)

    ylabel_str = make_ylabel(qps_mode, unit_s)
    if SHOW_TITLE:
        title_str = make_title_prefix(qps_mode, total_queries)
        fig.suptitle(f"{title_str} — {index_name},  {table},  SF={sf},  {system}",
                     fontsize=TITLE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT, y=SUPTITLE_Y_SUMMARY_GRID)
    fig.text(0.5, -0.01, "Batch Size", ha='center',
             fontsize=LABEL_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
    fig.text(-0.01, 0.5, ylabel_str, va='center', rotation='vertical',
             fontsize=LABEL_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)

    fig.subplots_adjust(**SUBPLOTS_ADJUST)

    safe_idx = make_safe_name(index_name)
    out_dir  = os.path.join(out_base, f"sf_{sf}", system, safe_idx)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{system}_sf{sf}_k{_K_VALUE}_{safe_idx}_summary_grid_{table}{_EXT}")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT C] {out_path}")


def plot_summary_grid(df, index_name, sf, system, total_queries, metric,
                      out_base, log_y, qps_mode, show_grid, show_values, unit_s=False,
                      show_values_minimal=False, ann_min_gap=15):
    tables = {}
    for q in sorted(df["query"].unique()):
        tables.setdefault(get_table(q), []).append(q)
    cases = [c for c in CASE_ORDER if c in df["case"].values]

    for table, queries in sorted(tables.items()):
        _plot_summary_grid_table(df, table, queries, cases, index_name, sf, system,
                                  total_queries, metric, out_base, log_y,
                                  qps_mode, show_grid, show_values, unit_s)


# ---------------------------------------------------------------------------
# Plot D: all cases on one figure per table (lines = cases, subplots = queries)
# ---------------------------------------------------------------------------

def plot_all_cases_per_table(df, index_name, sf, system, total_queries, metric,
                              out_base, log_y, qps_mode, show_grid, show_values, unit_s=False,
                              show_values_minimal=False, ann_min_gap=15):
    """One figure per (index, table): subplots = queries, lines = cases.

    Unlike the grid (rows=queries × cols=cases), here all cases appear in the
    same subplot so QPS across cases is directly comparable per query.
    """
    all_queries   = sorted(df["query"].unique())
    cases_present = [c for c in CASE_ORDER if c in df["case"].values]

    tables = {}
    for q in all_queries:
        tables.setdefault(get_table(q), []).append(q)

    for table, queries in sorted(tables.items()):
        n = len(queries)
        if n == 0:
            continue

        fig, axes = plt.subplots(1, n, figsize=(FIGSIZE_DOUBLE_W, FIGSIZE_H_PER_ROW), squeeze=False)

        for col, query in enumerate(queries):
            ax  = axes[0][col]
            all_ys = []

            for case in cases_present:
                sub = df[(df["query"] == query) & (df["case"] == case)].sort_values("batch_size")
                if sub.empty:
                    continue
                xs, ys, es = compute_series(sub, metric, total_queries, qps_mode, unit_s)
                all_ys.extend(ys.tolist())
                color  = CASE_COLORS[case]
                marker = CASE_MARKERS[case]

                ax.plot(xs, ys, color=color, marker=marker, label=CASE_RENAME_MAPPING[case],
                        linewidth=2, markersize=7,
                        markeredgecolor=EDGE_COLOR, markeredgewidth=0.6)
                _fill_std_band(ax, xs, ys, es, color)

            set_xaxis_batch(ax)
            if log_y:
                ax.set_yscale("log")
            if show_values:
                expand_ylim_for_labels(ax, all_ys, log_y)
            apply_grid(ax, show_grid)
            ax.tick_params(axis='both', labelsize=SUBPLOT_TICK_FONTSIZE)
            ax.set_xlabel("Batch Size", fontsize=SUBPLOT_LABEL_FONTSIZE,
                          fontweight=DEFAULT_FONT_WEIGHT, labelpad=XLABEL_PAD)
            ax.set_title(query, fontsize=SUBPLOT_TITLE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT, pad=TITLE_PAD)
            if col == 0:
                ax.set_ylabel(make_ylabel(qps_mode, unit_s), fontsize=SUBPLOT_LABEL_FONTSIZE,
                              fontweight=DEFAULT_FONT_WEIGHT, labelpad=YLABEL_PAD)
            ax.legend(fontsize=SUBPLOT_LEGEND_FONTSIZE, framealpha=0.9)

            if show_values:
                _ann_series = []
                for case in cases_present:
                    sub = df[(df["query"] == query) & (df["case"] == case)].sort_values("batch_size")
                    if sub.empty:
                        continue
                    xs, ys, _ = compute_series(sub, metric, total_queries, qps_mode, unit_s)
                    _ann_series.append((xs, ys))
                if show_values_minimal:
                    annotate_points_minimal(ax, _ann_series, log_y, qps_mode, unit_s, ann_min_gap)
                else:
                    for xs, ys in _ann_series:
                        annotate_points(ax, xs, ys, log_y, qps_mode, unit_s)

        if SHOW_TITLE:
            title_str = make_title_prefix(qps_mode, total_queries)
            fig.suptitle(f"{title_str} — {index_name},  {table},  SF={sf},  {system}",
                         fontsize=TITLE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT, y=SUPTITLE_Y)
        fig.tight_layout(pad=TIGHT_LAYOUT_PAD, rect=TIGHT_LAYOUT_RECT)

        safe_idx = make_safe_name(index_name)
        out_dir  = os.path.join(out_base, f"sf_{sf}", system, safe_idx)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{system}_sf{sf}_k{_K_VALUE}_{safe_idx}_all_cases_per_query_{table}{_EXT}")
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT D] {out_path}")


# ---------------------------------------------------------------------------
# Plot E: cross-index per (query, case) — lines = indexes
# ---------------------------------------------------------------------------

def plot_cross_index_per_query_case(df, sf, system, total_queries, metric,
                                    out_base, log_y, qps_mode, show_grid,
                                    show_values, unit_s=False, show_values_minimal=False, ann_min_gap=15):
    """One figure per (query, case, table): lines = indexes.

    Answers: 'For ann_reviews running GPU-GPU-GPU, how do Flat / IVF / Cagra compare?'
    """
    tables = {}
    for q in sorted(df["query"].unique()):
        tables.setdefault(get_table(q), []).append(q)
    cases_present = [c for c in CASE_ORDER if c in df["case"].values]
    indexes = sorted(df["index_name"].unique())

    for table, queries in sorted(tables.items()):
        for query in queries:
            for case in cases_present:
                sub = df[(df["query"] == query) & (df["case"] == case)]
                # Skip if no index has data for this (query, case)
                if all(sub[sub["index_name"] == idx].empty for idx in indexes):
                    continue

                fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
                all_ys = []

                for idx_name in indexes:
                    idf = sub[sub["index_name"] == idx_name].sort_values("batch_size")
                    if idf.empty:
                        continue
                    xs, ys, es = compute_series(idf, metric, total_queries, qps_mode, unit_s)
                    all_ys.extend(ys.tolist())
                    color  = INDEX_COLORS.get(idx_name, "#888888")
                    marker = INDEX_MARKERS.get(idx_name, "o")

                    ax.plot(xs, ys, color=color, marker=marker, label=get_index_display_name(idx_name),
                            linewidth=2, markersize=7,
                            markeredgecolor=EDGE_COLOR, markeredgewidth=0.6)
                    _fill_std_band(ax, xs, ys, es, color)

                case_label = CASE_RENAME_MAPPING[case]
                _finalize_ax(ax, log_y, show_values, show_grid, all_ys,
                             xlabel="Batch Size",
                             ylabel=make_ylabel(qps_mode, unit_s),
                             title=f"{make_title_prefix(qps_mode, total_queries)}\n"
                                   f"{query},  {case_label},  SF={sf},  {system}")

                if show_values:
                    _ann_series = []
                    for idx_name in indexes:
                        idf = sub[sub["index_name"] == idx_name].sort_values("batch_size")
                        if idf.empty:
                            continue
                        xs, ys, _ = compute_series(idf, metric, total_queries, qps_mode, unit_s)
                        _ann_series.append((xs, ys))
                    if show_values_minimal:
                        annotate_points_minimal(ax, _ann_series, log_y, qps_mode, unit_s, ann_min_gap)
                    else:
                        for xs, ys in _ann_series:
                            annotate_points(ax, xs, ys, log_y, qps_mode, unit_s)

                safe_case = make_safe_name(case)
                out_dir = os.path.join(out_base, f"sf_{sf}", system,
                                       "cross_index", "per_query_case", table)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{system}_sf{sf}_k{_K_VALUE}_cross_index_{query}_{safe_case}{_EXT}")
                fig.tight_layout(pad=TIGHT_LAYOUT_PAD)
                fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
                plt.close(fig)
                print(f"[PLOT E] {out_path}")


# ---------------------------------------------------------------------------
# Plot F: cross-index case grid per query — subplots=cases, lines=indexes
# ---------------------------------------------------------------------------

def plot_cross_index_case_grid(df, sf, system, total_queries, metric,
                               out_base, log_y, qps_mode, show_grid,
                               show_values, unit_s=False, show_values_minimal=False, ann_min_gap=15):
    """One figure per (query, table): one subplot per case, lines = indexes.

    Grid version of Plot E: all cases side-by-side so index scaling can be
    compared across hardware configs in a single image.
    """
    tables = {}
    for q in sorted(df["query"].unique()):
        tables.setdefault(get_table(q), []).append(q)
    cases_present = [c for c in CASE_ORDER if c in df["case"].values]
    indexes = sorted(df["index_name"].unique())

    n_cases = len(cases_present)
    # Layout: up to 2 columns, rows as needed
    n_cols = min(n_cases, 2)
    n_rows = math.ceil(n_cases / n_cols)

    for table, queries in sorted(tables.items()):
        for query in queries:
            qdf = df[df["query"] == query]
            if qdf.empty:
                continue

            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(FIGSIZE_DOUBLE_W, FIGSIZE_H_PER_ROW * n_rows),
                                     squeeze=False)

            for i, case in enumerate(cases_present):
                ax     = axes[i // n_cols][i % n_cols]
                all_ys = []

                for idx_name in indexes:
                    sub = qdf[(qdf["case"] == case) & (qdf["index_name"] == idx_name)
                              ].sort_values("batch_size")
                    if sub.empty:
                        continue
                    xs, ys, es = compute_series(sub, metric, total_queries, qps_mode, unit_s)
                    all_ys.extend(ys.tolist())
                    color  = INDEX_COLORS.get(idx_name, "#888888")
                    marker = INDEX_MARKERS.get(idx_name, "o")

                    ax.plot(xs, ys, color=color, marker=marker, label=get_index_display_name(idx_name),
                            linewidth=2, markersize=6,
                            markeredgecolor=EDGE_COLOR, markeredgewidth=0.6)
                    _fill_std_band(ax, xs, ys, es, color)

                case_label = CASE_RENAME_MAPPING[case]
                _finalize_ax(ax, log_y, show_values, show_grid, all_ys,
                             xlabel="Batch Size",
                             ylabel=make_ylabel(qps_mode, unit_s),
                             title=case_label,
                             unit_s=unit_s, qps_mode=qps_mode,
                             show_title=True)
                if i % n_cols != 0:
                    ax.set_ylabel("")

                if show_values:
                    _ann_series = []
                    for idx_name in indexes:
                        sub = qdf[(qdf["case"] == case) & (qdf["index_name"] == idx_name)
                                  ].sort_values("batch_size")
                        if sub.empty:
                            continue
                        xs, ys, _ = compute_series(sub, metric, total_queries, qps_mode, unit_s)
                        _ann_series.append((xs, ys))
                    if show_values_minimal:
                        annotate_points_minimal(ax, _ann_series, log_y, qps_mode, unit_s, ann_min_gap)
                    else:
                        for xs, ys in _ann_series:
                            annotate_points(ax, xs, ys, log_y, qps_mode, unit_s)

            # Hide any unused subplots (if n_cases is odd with 2-col layout)
            for i in range(n_cases, n_rows * n_cols):
                axes[i // n_cols][i % n_cols].set_visible(False)

            if SHOW_TITLE:
                title_str = make_title_prefix(qps_mode, total_queries)
                fig.suptitle(f"{title_str} — {query},  {table},  SF={sf},  {system}",
                             fontsize=TITLE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT, y=SUPTITLE_Y)
            fig.tight_layout(pad=TIGHT_LAYOUT_PAD, rect=TIGHT_LAYOUT_RECT)

            out_dir = os.path.join(out_base, f"sf_{sf}", system,
                                   "cross_index", "case_grid", table)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{system}_sf{sf}_k{_K_VALUE}_cross_index_case_grid_{query}{_EXT}")
            fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"[PLOT F] {out_path}")


# ---------------------------------------------------------------------------
# Plot G/H: operator breakdown stacked bars (--plot_operator_breakdown)
# ---------------------------------------------------------------------------

def _plot_operator_breakdown(df_op, index_name, stat, sf, system,
                              out_base, segments, labels, colors,
                              plot_tag, plot_label):
    """Stacked bar breakdown plot: one figure per (query, case, index).

    X-axis = batch_size (log scale, categorical), Y-axis = ms.
    Segments are the operator columns listed in `segments`.
    """
    queries = sorted(df_op["query"].unique())
    cases   = [c for c in CASE_ORDER if c in df_op["case"].values]

    for query in queries:
        for case in cases:
            sub = df_op[
                (df_op["query"] == query) &
                (df_op["case"]  == case)  &
                (df_op["stat"]  == stat)
            ].sort_values("batch_size")
            if sub.empty:
                continue

            batch_sizes = sorted(sub["batch_size"].unique())
            xs = [str(bs) for bs in batch_sizes]  # categorical x-axis

            # Compute derived columns for detailed breakdown (Plot 1)
            sub = sub.copy()
            if "RelOperators_ms" not in sub.columns:
                if "Operators_ms" in sub.columns and "VectorSearch_ms" in sub.columns:
                    sub["RelOperators_ms"] = (sub["Operators_ms"] - sub["VectorSearch_ms"]).clip(lower=0)
            if "DataMovement_ms" not in sub.columns:
                if "DataTransfers_ms" in sub.columns and "IndexMovement_ms" in sub.columns:
                    sub["DataMovement_ms"] = (sub["DataTransfers_ms"] - sub["IndexMovement_ms"]).clip(lower=0)

            # Build stacked data
            bottom = np.zeros(len(batch_sizes))
            fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

            for seg, lbl, col in zip(segments, labels, colors):
                if seg not in sub.columns:
                    continue
                heights = []
                for bs in batch_sizes:
                    row = sub[sub["batch_size"] == bs]
                    heights.append(float(row[seg].iloc[0]) if not row.empty else 0.0)
                heights = np.array(heights)
                ax.bar(xs, heights, bottom=bottom, label=lbl, color=col,
                       edgecolor="black", linewidth=0.4)
                bottom += heights

            ax.set_xlabel("Batch Size", fontsize=LABEL_FONTSIZE,
                          fontweight=DEFAULT_FONT_WEIGHT, labelpad=XLABEL_PAD)
            ax.set_ylabel("Time [ms]", fontsize=LABEL_FONTSIZE,
                          fontweight=DEFAULT_FONT_WEIGHT, labelpad=YLABEL_PAD)
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
            ax.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.9)

            case_label = CASE_RENAME_MAPPING.get(case, case)
            if SHOW_TITLE:
                ax.set_title(f"{plot_label} ({stat})\n{query},  {case_label},  "
                             f"{index_name},  SF={sf},  {system}",
                             fontsize=TITLE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT,
                             pad=TITLE_PAD)

            safe_idx  = make_safe_name(index_name)
            safe_case = make_safe_name(case)
            out_dir   = os.path.join(out_base, f"sf_{sf}", system, safe_idx,
                                     plot_tag, stat)
            os.makedirs(out_dir, exist_ok=True)
            out_path  = os.path.join(out_dir,
                                     f"{system}_sf{sf}_{safe_idx}_{plot_tag}_{query}_{safe_case}{_EXT}")
            fig.tight_layout(pad=TIGHT_LAYOUT_PAD)
            fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"[{plot_tag.upper()}] {out_path}")


def _plot_operator_breakdown_summary(df_op, index_name, stat, sf, system,
                                     out_base, segments, labels, colors,
                                     plot_tag, plot_label):
    """Summary grid: one figure per query, subplots = one per case.

    Each subplot is a stacked-bar chart (batch_size on X, ms on Y) for that
    case, laid out side-by-side so all cases for one query are visible at once.
    Legend is placed below the subplots to avoid overlap.
    """
    queries = sorted(df_op["query"].unique())
    cases   = [c for c in CASE_ORDER if c in df_op["case"].values]
    if not queries or not cases:
        return

    n_cases = len(cases)
    # One row of subplots (one per case); legend row added via subplots_adjust
    fig_w = max(3.2 * n_cases, 7.0)
    fig_h = 3.2   # subplot area; extra space for legend handled by subplots_adjust

    safe_idx = make_safe_name(index_name)

    for query in queries:
        fig, axes = plt.subplots(1, n_cases, figsize=(fig_w, fig_h), squeeze=False)
        axes_flat = axes.flatten()
        legend_handles, legend_lbls = [], []

        for ax_idx, case in enumerate(cases):
            ax  = axes_flat[ax_idx]
            sub = df_op[
                (df_op["query"] == query) &
                (df_op["case"]  == case)  &
                (df_op["stat"]  == stat)
            ].sort_values("batch_size")

            case_label = CASE_RENAME_MAPPING.get(case, case)
            ax.set_title(case_label, fontsize=SUBPLOT_TITLE_FONTSIZE,
                         fontweight=DEFAULT_FONT_WEIGHT, pad=TITLE_PAD)

            if sub.empty:
                ax.set_visible(False)
                continue

            # Compute derived columns for detailed breakdown (Plot 1)
            sub = sub.copy()
            if "RelOperators_ms" not in sub.columns:
                if "Operators_ms" in sub.columns and "VectorSearch_ms" in sub.columns:
                    sub["RelOperators_ms"] = (sub["Operators_ms"] - sub["VectorSearch_ms"]).clip(lower=0)
            if "DataMovement_ms" not in sub.columns:
                if "DataTransfers_ms" in sub.columns and "IndexMovement_ms" in sub.columns:
                    sub["DataMovement_ms"] = (sub["DataTransfers_ms"] - sub["IndexMovement_ms"]).clip(lower=0)

            batch_sizes = sorted(sub["batch_size"].unique())
            xs          = [str(bs) for bs in batch_sizes]
            bottom      = np.zeros(len(batch_sizes))

            for seg, lbl, col in zip(segments, labels, colors):
                if seg not in sub.columns:
                    continue
                heights = np.array([
                    float(sub[sub["batch_size"] == bs][seg].iloc[0])
                    if not sub[sub["batch_size"] == bs].empty else 0.0
                    for bs in batch_sizes
                ])
                bar = ax.bar(xs, heights, bottom=bottom, label=lbl, color=col,
                             edgecolor="black", linewidth=0.4)
                bottom += heights
                if not legend_handles:
                    pass  # will collect after loop

            ax.set_xlabel("Batch Size", fontsize=SUBPLOT_LABEL_FONTSIZE,
                          labelpad=XLABEL_PAD)
            ax.set_ylabel("Time [ms]", fontsize=SUBPLOT_LABEL_FONTSIZE,
                          labelpad=YLABEL_PAD)
            ax.tick_params(axis="both", labelsize=SUBPLOT_TICK_FONTSIZE)

            # Collect legend entries from first non-empty axis
            if not legend_handles:
                legend_handles, legend_lbls = ax.get_legend_handles_labels()

        if SHOW_TITLE:
            fig.suptitle(f"{plot_label} ({stat}) — {query}, {index_name}, SF={sf}, {system}",
                         fontsize=SUBPLOT_TITLE_FONTSIZE, y=1.01)

        # Legend below all subplots — reserve bottom space via subplots_adjust
        legend_ncol = min(len(legend_lbls), 5)
        if legend_handles:
            fig.legend(legend_handles, legend_lbls,
                       loc="lower center",
                       ncol=legend_ncol,
                       fontsize=SUBPLOT_LEGEND_FONTSIZE,
                       framealpha=0.9,
                       bbox_to_anchor=(0.5, 0.0))
        # Leave room at bottom for legend (≈0.08 per legend row)
        legend_rows = math.ceil(len(legend_lbls) / legend_ncol)
        bottom_pad  = 0.06 + 0.055 * legend_rows
        fig.tight_layout(pad=TIGHT_LAYOUT_PAD)
        fig.subplots_adjust(bottom=bottom_pad)

        safe_query = make_safe_name(query)
        out_dir    = os.path.join(out_base, f"sf_{sf}", system, safe_idx,
                                  plot_tag + "_summary", stat)
        os.makedirs(out_dir, exist_ok=True)
        out_path   = os.path.join(out_dir,
                                  f"{system}_sf{sf}_{safe_idx}_{plot_tag}_summary_{safe_query}{_EXT}")
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[{plot_tag.upper()}_SUMMARY] {out_path}")


def plot_operator_breakdown(df_op, index_name, sf, system, out_base, stats=None):
    """Plot both breakdown levels (Plot 1 and Plot 2) for all stats."""
    if stats is None:
        stats = ["median"]
    for stat in stats:
        _plot_operator_breakdown(
            df_op, index_name, stat, sf, system, out_base,
            BREAKDOWN1_SEGMENTS, BREAKDOWN1_LABELS, BREAKDOWN1_COLORS,
            "operator_breakdown_1", "High-Level Breakdown",
        )
        _plot_operator_breakdown_summary(
            df_op, index_name, stat, sf, system, out_base,
            BREAKDOWN1_SEGMENTS, BREAKDOWN1_LABELS, BREAKDOWN1_COLORS,
            "operator_breakdown_1", "High-Level Breakdown",
        )
        _plot_operator_breakdown(
            df_op, index_name, stat, sf, system, out_base,
            BREAKDOWN2_SEGMENTS, BREAKDOWN2_LABELS, BREAKDOWN2_COLORS,
            "operator_breakdown_2", "Operator-Level Breakdown",
        )
        _plot_operator_breakdown_summary(
            df_op, index_name, stat, sf, system, out_base,
            BREAKDOWN2_SEGMENTS, BREAKDOWN2_LABELS, BREAKDOWN2_COLORS,
            "operator_breakdown_2", "Operator-Level Breakdown",
        )


# ---------------------------------------------------------------------------
# Plot S: Summary — pre-filtering (Exhaustive) vs post-filtering (ANN) on one figure
# ---------------------------------------------------------------------------

def plot_summary_prepost(df, sf, system, total_queries, metric,
                         out_base, log_y, qps_mode, show_grid, show_values,
                         unit_s=False, show_values_minimal=False, ann_min_gap=15):
    """One figure per (table, selectivity variant).
    4 subplots: CPU | Mixed | Hybrid | GPU.

    Lines combine pre-filtering (Flat/Exhaustive) and post-filtering (ANN indexes)
    from different index CSVs, with paper-style prefixed names.

    Matching groups (each gets its own figure):
      base: pre_images <-> post_images <-> pre_images_hybrid <-> post_images_hybrid
      low:  pre_images_low_sel <-> post_images_low_sel <-> ..._hybrid_low_sel
      high: pre_images_high_sel <-> post_images_high_sel <-> ..._hybrid_high_sel
    """
    all_queries = set(df["query"].unique())
    indexes = sorted(df["index_name"].unique())

    def _color_for_label(lbl):
        """Map a line label like 'pre_Exhaustive' or 'post_CAGRA' to its index color."""
        core = lbl.split("_", 1)[1]  # drop pre_/post_ prefix
        for raw, disp in INDEX_DISPLAY_NAMES.items():
            if disp == core:
                return INDEX_COLORS.get(raw, "#888888")
        return "#888888"

    for table, variants in SUMMARY_QUERY_GROUPS.items():
        for sel_key, (pre_q, post_q) in variants.items():
            # Build line specs: (subplot_key, csv_query, csv_case, index_name, label)
            line_specs = []

            # --- Non-hybrid subplots (cases 0, 1, 3) ---
            for case_str, _ in SUMMARY_SUBPLOT_CASES:
                if case_str == "hybrid":
                    continue
                if pre_q in all_queries and "Flat" in indexes:
                    line_specs.append((case_str, pre_q, case_str, "Flat",
                                       "pre_Exhaustive"))
                if post_q in all_queries:
                    for idx in indexes:
                        if idx == "Flat":
                            continue
                        display = get_index_display_name(idx)
                        line_specs.append((case_str, post_q, case_str, idx,
                                           f"post_{display}"))

            # --- Hybrid subplot ---
            hybrid_pre_q = SUMMARY_HYBRID_MAP.get(pre_q)
            hybrid_post_q = SUMMARY_HYBRID_MAP.get(post_q)
            if hybrid_pre_q and hybrid_pre_q in all_queries and "Flat" in indexes:
                line_specs.append(("hybrid", hybrid_pre_q,
                                   SUMMARY_HYBRID_CASE["pre"], "Flat",
                                   "pre_Exhaustive"))
            if hybrid_post_q and hybrid_post_q in all_queries:
                for idx in indexes:
                    if idx == "Flat":
                        continue
                    display = get_index_display_name(idx)
                    line_specs.append(("hybrid", hybrid_post_q,
                                       SUMMARY_HYBRID_CASE["post"], idx,
                                       f"post_{display}"))

            if not line_specs:
                continue

            # Unique labels (preserving insertion order)
            seen_labels = []
            for *_, lbl in line_specs:
                if lbl not in seen_labels:
                    seen_labels.append(lbl)

            markers_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "p"]
            label_colors = {lbl: _color_for_label(lbl) for lbl in seen_labels}
            label_markers = {lbl: markers_cycle[i % len(markers_cycle)]
                             for i, lbl in enumerate(seen_labels)}
            label_linestyle = {lbl: "--" if lbl.startswith("pre_") else "-"
                               for lbl in seen_labels}

            # Build subplots: 1 row x 4 cols (wide, compact height)
            n_sub = len(SUMMARY_SUBPLOT_CASES)
            fig, axes = plt.subplots(1, n_sub,
                                     figsize=(FIGSIZE_DOUBLE_W * 1.6, FIGSIZE_H_PER_ROW),
                                     squeeze=False)

            legend_handles, legend_lbls = [], []

            for col, (case_key, case_title) in enumerate(SUMMARY_SUBPLOT_CASES):
                ax = axes[0][col]
                all_ys = []
                plotted = set()

                for sp_key, csv_q, csv_case, idx_name, label in line_specs:
                    if sp_key != case_key or label in plotted:
                        continue

                    sub = df[(df["query"] == csv_q) &
                             (df["index_name"] == idx_name) &
                             (df["case"] == csv_case)].sort_values("batch_size")
                    if sub.empty:
                        continue

                    xs, ys, es = compute_series(sub, metric, total_queries, qps_mode, unit_s)
                    all_ys.extend(ys.tolist())

                    color = label_colors[label]
                    marker = label_markers[label]
                    ls = label_linestyle[label]

                    ax.plot(xs, ys, color=color, marker=marker, linestyle=ls,
                            linewidth=1.8, markersize=5,
                            markeredgecolor=EDGE_COLOR, markeredgewidth=0.5)
                    _fill_std_band(ax, xs, ys, es, color)
                    plotted.add(label)

                    if label not in legend_lbls:
                        legend_lbls.append(label)
                        legend_handles.append(plt.Line2D(
                            [0], [0], color=color, marker=marker, linestyle=ls,
                            linewidth=1.8, markersize=5,
                            markeredgecolor=EDGE_COLOR, markeredgewidth=0.5))

                set_xaxis_batch(ax)
                if log_y:
                    ax.set_yscale("log")
                apply_grid(ax, show_grid)
                ax.tick_params(axis='both', labelsize=SUBPLOT_TICK_FONTSIZE)
                ax.set_title(case_title, fontsize=SUBPLOT_TITLE_FONTSIZE,
                             fontweight=DEFAULT_FONT_WEIGHT, pad=TITLE_PAD)
                ax.set_xlabel("Batch Size", fontsize=SUBPLOT_LABEL_FONTSIZE,
                              fontweight=DEFAULT_FONT_WEIGHT, labelpad=XLABEL_PAD)
                if col == 0:
                    ax.set_ylabel(make_ylabel(qps_mode, unit_s),
                                  fontsize=SUBPLOT_LABEL_FONTSIZE,
                                  fontweight=DEFAULT_FONT_WEIGHT, labelpad=YLABEL_PAD)

            sel_tag = "" if sel_key == "base" else f"_{sel_key}"
            if SHOW_TITLE:
                fig.suptitle(f"Pre/Post Summary — {table}{sel_tag},  SF={sf},  {system}",
                             fontsize=TITLE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT,
                             y=SUPTITLE_Y)

            fig.tight_layout(pad=TIGHT_LAYOUT_PAD, w_pad=2.0)
            legend_ncol = min(len(legend_lbls), 5)
            if legend_handles:
                fig.legend(legend_handles, legend_lbls,
                           loc="upper center", ncol=legend_ncol,
                           fontsize=SUBPLOT_LEGEND_FONTSIZE, framealpha=0.9,
                           bbox_to_anchor=(0.5, -0.02))

            out_dir = os.path.join(out_base, f"sf_{sf}", system,
                                   "cross_index", "summary")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(
                out_dir,
                f"{system}_sf{sf}_k{_K_VALUE}_summary_prepost_{table}{sel_tag}{_EXT}")
            fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"[PLOT S] {out_path}")


# ---------------------------------------------------------------------------
# Plot SO: Summary operator breakdown — stacked bars, same pre/post grouping
# ---------------------------------------------------------------------------

def plot_summary_prepost_operator(op_dfs, sf, system, out_base, stat="median",
                                  segments=None, labels=None, colors=None):
    """Grid figure per (table, selectivity variant).
    Rows = prefixed indexes (pre_Exhaustive, post_CAGRA, ...).
    Cols = cases (CPU, Mixed, Hybrid, GPU).
    Each cell = stacked bar chart with batch_size on x-axis.

    op_dfs: dict mapping index_name -> operator-breakdown DataFrame (unfiltered).
    """
    if segments is None:
        segments = BREAKDOWN2_SEGMENTS
    if labels is None:
        labels = BREAKDOWN2_LABELS
    if colors is None:
        colors = BREAKDOWN2_COLORS

    # Collect all queries across all index DataFrames
    all_queries = set()
    for op_df in op_dfs.values():
        all_queries.update(op_df["query"].unique())
    indexes = sorted(op_dfs.keys())

    for table, variants in SUMMARY_QUERY_GROUPS.items():
        for sel_key, (pre_q, post_q) in variants.items():
            # Build row specs: list of (row_label, index_name, query_per_case)
            # query_per_case maps case_key -> (csv_query, csv_case)
            row_specs = []

            # Pre-filtering: Flat only
            if "Flat" in op_dfs:
                qpc = {}
                for case_str, _ in SUMMARY_SUBPLOT_CASES:
                    if case_str == "hybrid":
                        hq = SUMMARY_HYBRID_MAP.get(pre_q)
                        if hq and hq in all_queries:
                            qpc["hybrid"] = (hq, SUMMARY_HYBRID_CASE["pre"])
                    else:
                        if pre_q in all_queries:
                            qpc[case_str] = (pre_q, case_str)
                if qpc:
                    row_specs.append(("pre_Exhaustive", "Flat", qpc))

            # Post-filtering: each ANN index
            for idx in indexes:
                if idx == "Flat":
                    continue
                display = get_index_display_name(idx)
                qpc = {}
                for case_str, _ in SUMMARY_SUBPLOT_CASES:
                    if case_str == "hybrid":
                        hq = SUMMARY_HYBRID_MAP.get(post_q)
                        if hq and hq in all_queries:
                            qpc["hybrid"] = (hq, SUMMARY_HYBRID_CASE["post"])
                    else:
                        if post_q in all_queries:
                            qpc[case_str] = (post_q, case_str)
                if qpc:
                    row_specs.append((f"post_{display}", idx, qpc))

            if not row_specs:
                continue

            n_rows = len(row_specs)
            n_cols = len(SUMMARY_SUBPLOT_CASES)
            fig_w = max(2.8 * n_cols, FIGSIZE_DOUBLE_W)
            fig_h = 2.0 * n_rows

            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(fig_w, fig_h),
                                     squeeze=False)

            legend_handles, legend_lbls = [], []

            for row_i, (row_label, idx_name, qpc) in enumerate(row_specs):
                op_df = op_dfs[idx_name]

                # Compute derived columns
                op_df = op_df.copy()
                if "RelOperators_ms" not in op_df.columns:
                    if "Operators_ms" in op_df.columns and "VectorSearch_ms" in op_df.columns:
                        op_df["RelOperators_ms"] = (
                            op_df["Operators_ms"] - op_df["VectorSearch_ms"]).clip(lower=0)
                if "DataMovement_ms" not in op_df.columns:
                    if "DataTransfers_ms" in op_df.columns and "IndexMovement_ms" in op_df.columns:
                        op_df["DataMovement_ms"] = (
                            op_df["DataTransfers_ms"] - op_df["IndexMovement_ms"]).clip(lower=0)

                for col_i, (case_key, case_title) in enumerate(SUMMARY_SUBPLOT_CASES):
                    ax = axes[row_i][col_i]

                    if case_key not in qpc:
                        ax.set_visible(False)
                        continue

                    csv_q, csv_case = qpc[case_key]
                    sub = op_df[
                        (op_df["query"] == csv_q) &
                        (op_df["case"] == csv_case) &
                        (op_df["stat"] == stat)
                    ].sort_values("batch_size")

                    if sub.empty:
                        ax.text(0.5, 0.5, "N/A", ha='center', va='center',
                                transform=ax.transAxes, fontsize=TICK_FONTSIZE,
                                color='gray')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    batch_sizes = sorted(sub["batch_size"].unique())
                    xs = [str(bs) for bs in batch_sizes]
                    bottom = np.zeros(len(batch_sizes))

                    for seg, lbl, col in zip(segments, labels, colors):
                        if seg not in sub.columns:
                            continue
                        heights = np.array([
                            float(sub[sub["batch_size"] == bs][seg].iloc[0])
                            if not sub[sub["batch_size"] == bs].empty else 0.0
                            for bs in batch_sizes
                        ])
                        ax.bar(xs, heights, bottom=bottom, color=col,
                               edgecolor="black", linewidth=0.3, width=0.7)
                        bottom += heights

                        if not legend_handles or lbl not in legend_lbls:
                            if lbl not in legend_lbls:
                                legend_lbls.append(lbl)
                                from matplotlib.patches import Patch
                                legend_handles.append(Patch(facecolor=col,
                                                            edgecolor="black",
                                                            linewidth=0.3,
                                                            label=lbl))

                    ax.tick_params(axis="both", labelsize=SUBPLOT_TICK_FONTSIZE)
                    if row_i == 0:
                        ax.set_title(case_title, fontsize=SUBPLOT_TITLE_FONTSIZE,
                                     fontweight=DEFAULT_FONT_WEIGHT, pad=TITLE_PAD)
                    if col_i == 0:
                        ax.set_ylabel(f"{row_label}\n[ms]",
                                      fontsize=SUBPLOT_LABEL_FONTSIZE,
                                      fontweight=DEFAULT_FONT_WEIGHT, labelpad=YLABEL_PAD)
                    if row_i == n_rows - 1:
                        ax.set_xlabel("Batch Size", fontsize=SUBPLOT_LABEL_FONTSIZE,
                                      labelpad=XLABEL_PAD)

            sel_tag = "" if sel_key == "base" else f"_{sel_key}"
            if SHOW_TITLE:
                fig.suptitle(
                    f"Operator Breakdown — {table}{sel_tag},  SF={sf},  {system}",
                    fontsize=TITLE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT,
                    y=1.02)

            fig.tight_layout(pad=TIGHT_LAYOUT_PAD, h_pad=1.0, w_pad=1.5)
            legend_ncol = min(len(legend_lbls), 6)
            if legend_handles:
                fig.legend(legend_handles, legend_lbls,
                           loc="upper center", ncol=legend_ncol,
                           fontsize=SUBPLOT_LEGEND_FONTSIZE, framealpha=0.9,
                           bbox_to_anchor=(0.5, -0.02))

            out_dir = os.path.join(out_base, f"sf_{sf}", system,
                                   "cross_index", "summary")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(
                out_dir,
                f"{system}_sf{sf}_k{_K_VALUE}_summary_prepost_operators_{table}{sel_tag}{_EXT}")
            fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"[PLOT SO] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

KNOWN_INDEX_MAP = {
    "IVF1024_Flat":                  "IVF1024,Flat",
    "IVF1024(H)_Flat":               "IVF1024(H),Flat",
    "IVF1024_PQ8":                   "IVF1024,PQ8",
    "IVF4096_Flat":                  "IVF4096,Flat",
    "IVF4096(H)_Flat":               "IVF4096(H),Flat",
    "HNSW32_Flat":                   "HNSW32,Flat",
    "Cagra_64_32_NN_DESCENT":        "Cagra,64,32,NN_DESCENT",
    "Cagra(C+H)_64_32_NN_DESCENT":  "Cagra(C+H),64,32,NN_DESCENT",
    "Flat":                          "Flat",
}


# ---------------------------------------------------------------------------
# bs1_fullsweep (CASE 6): cumulative time vs. query index
# ---------------------------------------------------------------------------

# H / C+H variants share color with their baseline so pairs read as one group.
# Linestyle distinguishes inside the pair.
BS1_PAIR_BASELINE = {
    "IVF1024(H),Flat":             "IVF1024,Flat",
    "IVF4096(H),Flat":             "IVF4096,Flat",
    "Cagra(C+H),64,32,NN_DESCENT": "Cagra,64,32,NN_DESCENT",
}

# Combined CPU+GPU plot uses linestyle to encode device (and H/CH variant):
#   CPU baseline:  dotted   (no H variants exist on CPU)
#   GPU baseline:  solid
#   GPU (H)/(C+H): dashed
def _bs1_color_style(idx_name, is_cpu=False):
    """(color, linestyle) for bs1_fullsweep plotting. Color groups same index family."""
    base = BS1_PAIR_BASELINE.get(idx_name)
    if base is not None:
        # H / (C+H) variant — share baseline color, dashed
        return INDEX_COLORS.get(base, "#666666"), (0, (5, 2))
    color = INDEX_COLORS.get(idx_name, "#666666")
    if is_cpu:
        return color, (0, (1, 1.5))   # dotted for CPU
    return color, "-"                  # solid for GPU baseline


def _humanize_count(x, _pos=None):
    """k/M/G suffix formatter for matplotlib FuncFormatter. Max 1 decimal place."""
    def _trim(v):
        # 1 decimal, then strip trailing .0 — '1.2k', '510', '1k', '12k'
        return f"{v:.1f}".rstrip("0").rstrip(".")
    ax = abs(x)
    if ax >= 1e9: return f"{_trim(x/1e9)}G"
    if ax >= 1e6: return f"{_trim(x/1e6)}M"
    if ax >= 1e3: return f"{_trim(x/1e3)}k"
    if ax == 0:   return "0"
    if ax >= 1:   return f"{int(round(x))}" if x == int(x) else f"{x:.1f}".rstrip("0").rstrip(".")
    return f"{x:.2g}"


def _bs1_compute_series(qdf, metric_col, extrapolate_to):
    """Build {idx_name -> (xs_full, cum_full, n_real, move_ms, steady_ms, per_query_arr)}.
    cum_full is in ms (caller handles unit). xs_full is 1-based query count.
    Extrapolation appends geometric points up to extrapolate_to with steady_ms slope.
    """
    out = {}
    for idx_name, sub in qdf.groupby("index_name"):
        sub = sub.sort_values("qstart")
        xs = sub["qstart"].to_numpy()
        ys_per = sub[metric_col].to_numpy(dtype=float)
        n_real = len(xs)
        move_vals = sub["setup_index_movement_ms"].dropna().unique()
        move_ms = float(move_vals[0]) if len(move_vals) else 0.0
        contrib = ys_per.copy()
        contrib[0] = contrib[0] + move_ms
        cum_real = np.cumsum(contrib)
        xs_real = xs.astype(float) + 1.0
        # Exclude point 0 from steady-state: first query after move can be a cold outlier.
        # Median is robust to remaining outliers.
        steady_src = ys_per[1:] if len(ys_per) > 1 else ys_per
        steady_ms = float(np.median(steady_src))
        if extrapolate_to and extrapolate_to > n_real:
            last_count = float(n_real)
            future_x = np.geomspace(last_count + 1.0, float(extrapolate_to), num=120)
            future_cum = cum_real[-1] + (future_x - last_count) * steady_ms
            xs_full = np.concatenate([xs_real, future_x])
            cum_full = np.concatenate([cum_real, future_cum])
        else:
            xs_full, cum_full = xs_real, cum_real
        out[idx_name] = (xs_full, cum_full, n_real, move_ms, steady_ms, ys_per)
    return out


def _bs1_find_crossover(xs_b, cum_b, xs_h, cum_h):
    """Smallest x at which cum_h overtakes cum_b (i.e. baseline becomes faster).
    Returns (x_cross, y_cross) or (None, None). Assumes xs_b == xs_h elementwise."""
    if len(xs_b) != len(xs_h):
        return None, None
    diff = cum_b - cum_h   # >0 → baseline still slower (move dominates)
    for i in range(len(diff)):
        if diff[i] <= 0:
            return float(xs_b[i]), float(cum_b[i])
    return None, None


def _bs1_apply_axis_format(ax, log_x, log_y, show_grid, unit_label,
                           xlabel_extra="", title=None, font_scale=1.0):
    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_humanize_count))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_humanize_count))
    ax.set_xlabel("Query count" + (f" {xlabel_extra}" if xlabel_extra else ""),
                  fontsize=LABEL_FONTSIZE * font_scale,
                  fontweight=DEFAULT_FONT_WEIGHT)
    if show_grid:
        # Subtle: major lines only, low alpha (was minor+major loud dashed lines)
        ax.grid(True, which="major", linestyle="-", linewidth=0.4,
                color="#cccccc", alpha=0.6, zorder=0)
        ax.grid(False, which="minor")
    else:
        ax.grid(False)
    ax.tick_params(labelsize=TICK_FONTSIZE * font_scale)
    if title and SHOW_TITLE:
        ax.set_title(title, fontsize=TITLE_FONTSIZE * font_scale,
                     fontweight=DEFAULT_FONT_WEIGHT)


def plot_bs1_fullsweep(combined_df, sf, system, out_base,
                       metric="mean", unit_s=False, log_y=False, log_x=False,
                       show_grid=True, extrapolate_to=1_000_000,
                       style="wide", device_filter=None):
    """Cumulative total time vs query count (one figure per query).

    - First point includes the one-time setup_index_movement.
    - Beyond N real queries, extrapolates linearly at steady-state per-query latency.
    - H / C+H variants share a color with their baseline (linestyle distinguishes them).
    - Crossover (where baseline catches up to its H-pair) annotated with a vertical line.
    - X and Y axes use k/M suffix tick labels.
    """
    df = combined_df.copy()
    if "qstart" not in df.columns or df["qstart"].isna().all():
        print("[bs1_fullsweep] No qstart data — skipping plot.")
        return
    df = df[df["qstart"].notna()].copy()
    df["qstart"] = df["qstart"].astype(int)
    metric_col = f"{metric}_ms"
    if metric_col not in df.columns:
        print(f"[bs1_fullsweep] metric column '{metric_col}' missing — skipping.")
        return

    # GPU-only filter: drop CPU case (0).
    if device_filter == "gpu":
        df = df[df["case"] != "0: CPU-CPU-CPU"].copy()
        if df.empty:
            print("[bs1_fullsweep] No GPU rows after filter — skipping.")
            return
    device_tag = "gpu_only" if device_filter == "gpu" else "combined"

    # Style presets: paper = single-column, compact, short labels, PDF; wide = readable
    if style == "paper":
        out_dir = os.path.join(out_base, "bs1_fullsweep")
        prefix = f"paper_{device_tag}_"
        figsize = FIGSIZE_SINGLE
        font_scale = 0.85
        ext = ".pdf"
        show_move_annot = False
        show_crossover_label = True     # succinct labels (see _short_xover_label)
        short_xover_label = True
        show_grid_eff = False           # paper convention: clean background
        line_width = 1.0
        marker_size = 14
    else:  # wide
        out_dir = os.path.join(out_base, "bs1_fullsweep")
        prefix = f"{device_tag}_"
        figsize = (FIGSIZE_DOUBLE_W, FIGSIZE_H_PER_ROW * 1.3)
        font_scale = 1.0
        ext = _EXT
        show_move_annot = True
        show_crossover_label = True
        short_xover_label = False
        show_grid_eff = show_grid
        line_width = 1.4
        marker_size = 24
    os.makedirs(out_dir, exist_ok=True)
    div = 1000.0 if unit_s else 1.0
    unit_label = "s" if unit_s else "ms"

    # Combined plot: all cases (CPU + GPU) overlaid per query.
    # Use a composite key (case, index) so identical index names (e.g. "Flat") from
    # different cases don't collide.
    for query, qdf in df.groupby("query"):
        series_combined = {}      # (case_label, idx_name) -> series tuple
        for case_label, ccdf in qdf.groupby("case"):
            sub_series = _bs1_compute_series(ccdf, metric_col, extrapolate_to)
            for idx_name, vals in sub_series.items():
                series_combined[(case_label, idx_name)] = vals
        if not series_combined:
            continue

        fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

        # Group legend by index family (Flat / IVF1024 / IVF4096 / Cagra), then
        # within family by (baseline → H/CH, CPU → GPU). Families ordered by
        # cheapest end-of-sweep total so fastest family appears first.
        def _family_and_variant(idx_name):
            if "Cagra(C+H)" in idx_name: return ("Cagra", 1)
            if "Cagra"      in idx_name: return ("Cagra", 0)
            if "IVF1024(H)" in idx_name: return ("IVF1024", 1)
            if "IVF1024"    in idx_name: return ("IVF1024", 0)
            if "IVF4096(H)" in idx_name: return ("IVF4096", 1)
            if "IVF4096"    in idx_name: return ("IVF4096", 0)
            if "Flat"       in idx_name: return ("Flat", 0)
            return (idx_name, 0)
        family_rank = {}
        for (case_label, idx_name), vals in series_combined.items():
            fam, _ = _family_and_variant(idx_name)
            end_cum = vals[1][-1]
            if fam not in family_rank or end_cum < family_rank[fam]:
                family_rank[fam] = end_cum
        def _sort_key(k):
            case_label, idx_name = k
            fam, variant = _family_and_variant(idx_name)
            is_gpu = 0 if case_label == "0: CPU-CPU-CPU" else 1
            return (family_rank[fam], fam, variant, is_gpu)
        keys_order = sorted(series_combined.keys(), key=_sort_key)

        for (case_label, idx_name) in keys_order:
            xs_full, cum_ms, n_real, move_ms, steady_ms, _ = series_combined[(case_label, idx_name)]
            cum = cum_ms / div
            is_cpu = (case_label == "0: CPU-CPU-CPU")
            color, linestyle = _bs1_color_style(idx_name, is_cpu=is_cpu)
            device_tag = "CPU" if is_cpu else "GPU"
            label = f"{get_index_display_name(idx_name)} ({device_tag})"
            ax.plot(xs_full, cum, color=color, linewidth=line_width, alpha=0.95,
                    linestyle=linestyle, label=label)
            ax.scatter(xs_full[:1], cum[:1], color=color, marker="o", s=marker_size,
                       edgecolors=EDGE_COLOR, linewidths=0.5, zorder=5)
            # Move annotations only meaningful for GPU; skip in combined view
            # (too crowded with 11 indexes).
            _ = (show_move_annot, move_ms)

        # Crossover annotations.
        # (a) GPU H/CH pairs: when the heavier-move baseline overtakes its (H)/(C+H) pair.
        # (b) CPU vs GPU: when the fastest GPU variant of a family overtakes the CPU one.
        # Definition: "X wins @ N" = from query N onward, X has lower cumulative time.
        cpu_case = "0: CPU-CPU-CPU"
        gpu_case = "1: GPU-CPU-CPU"
        crossover_idx = 0

        # (a) GPU H/CH crossovers
        for h_name, base_name in BS1_PAIR_BASELINE.items():
            k_h = (gpu_case, h_name)
            k_b = (gpu_case, base_name)
            if k_h not in series_combined or k_b not in series_combined:
                continue
            xs_b, cum_b, _, _, _, _ = series_combined[k_b]
            xs_h, cum_h, _, _, _, _ = series_combined[k_h]
            xc, yc = _bs1_find_crossover(xs_b, cum_b, xs_h, cum_h)
            if xc is None:
                continue
            color, _ = _bs1_color_style(base_name, is_cpu=False)
            ax.axvline(xc, color=color, linestyle=":", linewidth=0.9,
                       alpha=0.35, zorder=0.5)
            if show_crossover_label:
                base_disp = get_index_display_name(base_name)
                if short_xover_label:
                    label = f"{base_disp}<(H)"
                else:
                    label = f"{base_disp} (GPU) wins @ {_humanize_count(int(round(xc)))}"
                y_frac = 0.04 + 0.07 * crossover_idx
                ax.text(xc, y_frac, label,
                        transform=ax.get_xaxis_transform(),
                        fontsize=(VALUE_FONTSIZE) * font_scale,
                        color=color, alpha=0.95,
                        fontweight=DEFAULT_FONT_WEIGHT,
                        ha="left", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="white", edgecolor="none", alpha=0.85))
            crossover_idx += 1

        # (b) CPU vs GPU crossovers — for each CPU index, find the FIRST GPU variant
        # of the same family to overtake it (CPU starts low, GPU starts high due to
        # move; if GPU per-query is faster, eventually cum_gpu <= cum_cpu).
        # If no GPU variant ever wins, no annotation (CPU stays best).
        CPU_TO_GPU_FAMILY = {
            "Flat":                     ["Flat"],
            "IVF1024,Flat":             ["IVF1024,Flat", "IVF1024(H),Flat"],
            "IVF4096,Flat":             ["IVF4096,Flat", "IVF4096(H),Flat"],
            "Cagra,64,32,NN_DESCENT":   ["Cagra,64,32,NN_DESCENT",
                                         "Cagra(C+H),64,32,NN_DESCENT"],
        }
        for cpu_name, gpu_variants in CPU_TO_GPU_FAMILY.items():
            k_cpu = (cpu_case, cpu_name)
            if k_cpu not in series_combined:
                continue
            xs_c, cum_c, _, _, _, _ = series_combined[k_cpu]
            best_xc, best_yc, best_gpu = None, None, None
            for gpu_name in gpu_variants:
                k_gpu = (gpu_case, gpu_name)
                if k_gpu not in series_combined:
                    continue
                xs_g, cum_g, _, _, _, _ = series_combined[k_gpu]
                # smallest N where cum_g <= cum_c (GPU overtakes CPU)
                xc, yc = _bs1_find_crossover(xs_g, cum_g, xs_c, cum_c)
                if xc is not None and (best_xc is None or xc < best_xc):
                    best_xc, best_yc, best_gpu = xc, yc, gpu_name
            if best_xc is None:
                continue
            color, _ = _bs1_color_style(best_gpu, is_cpu=False)
            ax.axvline(best_xc, color=color, linestyle=":", linewidth=0.9,
                       alpha=0.35, zorder=0.5)
            if show_crossover_label:
                gpu_disp = get_index_display_name(best_gpu)
                if short_xover_label:
                    label = f"{gpu_disp}<CPU"
                else:
                    label = f"{gpu_disp} (GPU) wins (vs CPU) @ {_humanize_count(int(round(best_xc)))}"
                y_frac = 0.04 + 0.07 * crossover_idx
                ax.text(best_xc, y_frac, label,
                        transform=ax.get_xaxis_transform(),
                        fontsize=(VALUE_FONTSIZE) * font_scale,
                        color=color, alpha=0.95,
                        fontweight=DEFAULT_FONT_WEIGHT,
                        ha="left", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="white", edgecolor="none", alpha=0.85))
            crossover_idx += 1

        ax.set_ylabel(f"Cumulative time [{unit_label}]",
                      fontsize=LABEL_FONTSIZE * font_scale,
                      fontweight=DEFAULT_FONT_WEIGHT)
        ttl = (f"bs1_fullsweep — {query} (sf={sf}, {system})"
               f"{' [extrapolated to ' + _humanize_count(extrapolate_to) + ']' if extrapolate_to else ''}")
        _bs1_apply_axis_format(ax, log_x, log_y, show_grid_eff, unit_label,
                               title=ttl, font_scale=font_scale)
        ax.legend(fontsize=LEGEND_FONTSIZE * font_scale, loc="upper left",
                  frameon=(style == "wide"), ncol=2)

        fig.tight_layout()
        safe_q = make_safe_name(query)
        scale_tag = ("loglog" if (log_x and log_y) else
                     "logx"   if log_x else
                     "logy"   if log_y else "lin")
        out_path = os.path.join(out_dir,
                                f"{prefix}{safe_q}_sf_{sf}_{system}_{scale_tag}{ext}")
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        # Paper PDFs also get a PNG preview saved alongside (for in-chat inspection).
        if style == "paper" and ext == ".pdf":
            png_path = out_path[:-4] + ".png"
            fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
            print(f"  [SAVED] {png_path}  (preview)")
        plt.close(fig)
        print(f"  [SAVED] {out_path}")




def main():
    parser = argparse.ArgumentParser(description="Plot vary-batch results (default: QPS).")
    parser.add_argument("--sf",            type=str,  default="1")
    parser.add_argument("--system",        type=str,  default="sgs-gpu05")
    parser.add_argument("--total_queries", type=int,  default=10000,
                        help="Total queries for scaling (only used with --total_time)")
    parser.add_argument("--in_dir",        type=str,  default="./parse_caliper")
    parser.add_argument("--out_dir",       type=str,  default="plots")
    parser.add_argument("--result_dir",    type=str,  default="other-vary-batch-in-maxbench",
                        help="Result directory (relative to in_dir/out_dir). "
                             "Default: other-vary-batch-in-maxbench")
    parser.add_argument("--metric",        type=str,  default="median",
                        choices=["mean", "min", "median", "max"])
    parser.add_argument("--log_y",         action="store_true",
                        help="Log scale on Y axis")
    parser.add_argument("--total_time",    action="store_true",
                        help="Plot estimated total time instead of QPS (default: QPS)")
    parser.add_argument("--no_grid",       action="store_true",
                        help="Disable background grid lines")
    parser.add_argument("--show_values",   action="store_true",
                        help="Annotate each data point with its numeric value")
    parser.add_argument("--show_values_minimal", action="store_true",
                        help="When annotating values, show only the highest value per x "
                             "position (avoids overlap in multi-series plots)")
    parser.add_argument("--ann_min_gap", type=float, default=30,
                        help="With --show_values_minimal: minimum vertical pixel gap required "
                             "between two labels at the same x for both to be shown. The top "
                             "value is always shown; lower values are suppressed if closer than "
                             "this many pixels. 0=show all, 30=default, larger=more aggressive "
                             "suppression.")
    parser.add_argument("--skip_cases",    type=str,  default="2",
                        help="Comma-separated case numbers to skip (default: 2). "
                             "E.g. '2' to skip GPU-CPU-GPU, '0,1,2' to keep only case 3, "
                             "'' to show all cases")
    parser.add_argument("--no-std-band",   action="store_true",
                        help="Hide std-deviation shaded bands around lines (default: shown)")
    parser.add_argument("--no-per-query",  action="store_true",
                        help="Skip per-query individual plots (Plot A)")
    parser.add_argument("--title",         action="store_true",
                        help="Show plot titles (default: off)")
    parser.add_argument("--unit",          type=str,  default="ms", choices=["ms", "s"],
                        help="Y-axis unit for total_time mode: 'ms' (default) or 's'")
    parser.add_argument("--format",        type=str,  default="jpeg",
                        choices=["jpeg", "pdf", "png", "svg"],
                        help="Output image format. Use 'pdf' for vector quality in LaTeX papers.")
    parser.add_argument("--k",             type=str,  default="100",
                        help="k (number of nearest neighbors) label for filenames.")
    parser.add_argument("--plot_operator_breakdown", action="store_true",
                        help="Plot per-operator stacked bars from operator-breakdown CSVs "
                             "(requires parse_varbatch.py --parse_caliper to have run first)")
    parser.add_argument("--plot_bs1_fullsweep", action="store_true",
                        help="Plot bs1_fullsweep (CASE 6) cumulative total time vs qstart "
                             "(one line per index variant). Expects --result_dir=other-bs1_fullsweep.")
    parser.add_argument("--bs1_extrapolate_to", type=int, default=10_000,
                        help="bs1_fullsweep: extrapolate cumulative time forward up to this "
                             "many queries assuming steady-state per-query latency. "
                             "Default 1_000_000. Pass 0 to disable extrapolation.")
    parser.add_argument("--log_x", action="store_true",
                        help="Log scale on X axis (useful for bs1_fullsweep extrapolation).")
    parser.add_argument("--breakdown_stats", type=str, default="median",
                        help="Comma-separated stats to plot for breakdown (default: mean). "
                             "E.g. 'mean,median'")
    args = parser.parse_args()

    global _EXT, _K_VALUE, SHOW_TITLE, SHOW_STD_BAND, _PER_QUERY
    _EXT          = f".{args.format}"
    _K_VALUE      = args.k
    SHOW_TITLE    = args.title
    SHOW_STD_BAND = not args.no_std_band
    _PER_QUERY    = not args.no_per_query

    qps_mode   = not args.total_time
    unit_s     = (args.unit == "s")
    show_grid  = not args.no_grid
    show_values = args.show_values or args.show_values_minimal
    skip_cases = {f"{n}: " + {"0":"CPU-CPU-CPU","1":"GPU-CPU-CPU",
                               "2":"GPU-CPU-GPU","3":"GPU-GPU-GPU"}[n]
                  for n in args.skip_cases.split(",") if n.strip()
                  and n.strip() in ("0","1","2","3")}

    safe_result_dir = args.result_dir.replace("/", "_").replace("\\", "_")
    csv_dir   = os.path.join(args.in_dir, args.result_dir)
    pattern   = os.path.join(csv_dir,
                             f"{args.system}_{safe_result_dir}_*_sf_{args.sf}.csv")
    csv_files = [f for f in sorted(glob.glob(pattern))
                 if "_recall_" not in os.path.basename(f)]

    if not csv_files:
        print(f"[ERROR] No CSVs found matching: {pattern}")
        print("Run parse_varbatch.py first.")
        return

    out_base = os.path.join(args.out_dir, args.result_dir)

    all_dfs = []              # accumulated for cross-index plots (case-filtered)
    all_dfs_unfiltered = []   # unfiltered — summary plot needs case 2 for hybrid

    for csv_path in csv_files:
        fname      = os.path.basename(csv_path)
        prefix     = f"{args.system}_{safe_result_dir}_"
        suffix     = f"_sf_{args.sf}.csv"
        safe_index = fname[len(prefix):-len(suffix)]
        index_name = KNOWN_INDEX_MAP.get(safe_index, safe_index)

        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"[SKIP] Empty CSV: {csv_path}")
            continue

        if skip_cases:
            df = df[~df["case"].isin(skip_cases)]
        if df.empty:
            print(f"[SKIP] No data left after filtering cases for {index_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Plotting index: {index_name}  (file: {fname})")
        mode_str = "QPS" if qps_mode else f"TotalTime({args.total_queries}q)"
        print(f"  mode={mode_str}  show_values={show_values}  "
              f"show_values_minimal={args.show_values_minimal}  "
              f"grid={show_grid}  log_y={args.log_y}  skip_cases={skip_cases or 'none'}")
        print(f"{'='*60}")

        # bs1_fullsweep: skip normal batch-sweep plots (all rows are batch=1); handled below.
        if not args.plot_bs1_fullsweep:
            kwargs = dict(
                df=df, index_name=index_name, sf=args.sf, system=args.system,
                total_queries=args.total_queries, metric=args.metric,
                out_base=out_base, log_y=args.log_y,
                qps_mode=qps_mode, show_grid=show_grid, show_values=show_values,
                unit_s=unit_s, show_values_minimal=args.show_values_minimal,
                ann_min_gap=args.ann_min_gap,
            )
            if _PER_QUERY:
                plot_per_query(**kwargs)
            plot_per_case(**kwargs)
            plot_summary_grid(**kwargs)
            plot_all_cases_per_table(**kwargs)

        # Collect for cross-index plots (add index_name column)
        df_tagged = df.copy()
        df_tagged["index_name"] = index_name
        all_dfs.append(df_tagged)

        # Also collect unfiltered data for summary plot (needs case 2 for hybrid)
        df_raw = pd.read_csv(csv_path)
        df_raw["index_name"] = index_name
        all_dfs_unfiltered.append(df_raw)

    # -----------------------------------------------------------------------
    # Cross-index plots (E, F) — require data from all indexes combined
    # -----------------------------------------------------------------------
    if all_dfs and not args.plot_bs1_fullsweep:
        combined = pd.concat(all_dfs, ignore_index=True)
        cross_kwargs = dict(
            df=combined, sf=args.sf, system=args.system,
            total_queries=args.total_queries, metric=args.metric,
            out_base=out_base, log_y=args.log_y,
            qps_mode=qps_mode, show_grid=show_grid, show_values=show_values,
            unit_s=unit_s, show_values_minimal=args.show_values_minimal,
            ann_min_gap=args.ann_min_gap,
        )
        print(f"\n{'='*60}")
        print("Generating cross-index plots (E, F)...")
        print(f"{'='*60}")
        plot_cross_index_per_query_case(**cross_kwargs)
        plot_cross_index_case_grid(**cross_kwargs)

        print(f"\n{'='*60}")
        print("Generating summary pre/post plots (S)...")
        print(f"{'='*60}")
        # Summary plot needs unfiltered data (case 2 holds post_*_hybrid queries)
        combined_unfiltered = pd.concat(all_dfs_unfiltered, ignore_index=True)
        summary_kwargs = dict(cross_kwargs)
        summary_kwargs["df"] = combined_unfiltered
        plot_summary_prepost(**summary_kwargs)

    # -----------------------------------------------------------------------
    # bs1_fullsweep (CASE 6): cumulative time vs qstart
    # -----------------------------------------------------------------------
    if args.plot_bs1_fullsweep and all_dfs:
        print(f"\n{'='*60}")
        print("Generating bs1_fullsweep cumulative-time plots (loglog only)...")
        print(f"{'='*60}")
        combined_bs1 = pd.concat(all_dfs, ignore_index=True)
        # WIDE (interactive): loglog only. CLI --log_x/--log_y override either axis.
        log_x_v = True if not (args.log_x or args.log_y) else args.log_x
        log_y_v = True if not (args.log_x or args.log_y) else args.log_y
        # Two default variants per style: combined (CPU+GPU) and GPU-only.
        for dev_filter in (None, "gpu"):
            plot_bs1_fullsweep(
                combined_bs1, sf=args.sf, system=args.system,
                out_base=out_base, metric=args.metric,
                unit_s=unit_s, log_y=log_y_v, log_x=log_x_v, show_grid=show_grid,
                extrapolate_to=args.bs1_extrapolate_to, style="wide",
                device_filter=dev_filter,
            )
            plot_bs1_fullsweep(
                combined_bs1, sf=args.sf, system=args.system,
                out_base=out_base, metric=args.metric,
                unit_s=unit_s, log_y=True, log_x=True, show_grid=False,
                extrapolate_to=args.bs1_extrapolate_to, style="paper",
                device_filter=dev_filter,
            )

    # -----------------------------------------------------------------------
    # Operator breakdown plots (--plot_operator_breakdown)
    # -----------------------------------------------------------------------
    if args.plot_operator_breakdown:
        breakdown_stats = [s.strip() for s in args.breakdown_stats.split(",") if s.strip()]
        op_pattern = os.path.join(
            args.in_dir, args.result_dir,
            f"operator_breakdown_{args.system}_{safe_result_dir}_*_sf_{args.sf}.csv",
        )
        op_csv_files = sorted(glob.glob(op_pattern))
        if not op_csv_files:
            print(f"[WARN] No operator-breakdown CSVs found matching: {op_pattern}")
            print("       Run parse_varbatch.py --parse_caliper first.")
        else:
            print(f"\n{'='*60}")
            print("Generating operator breakdown plots (G/H)...")
            print(f"{'='*60}")
            op_dfs_unfiltered = {}  # for summary operator plot (needs case 2)
            for op_csv in op_csv_files:
                op_fname      = os.path.basename(op_csv)
                op_prefix     = f"operator_breakdown_{args.system}_{safe_result_dir}_"
                op_suffix     = f"_sf_{args.sf}.csv"
                op_safe_index = op_fname[len(op_prefix):-len(op_suffix)]
                op_index_name = KNOWN_INDEX_MAP.get(op_safe_index, op_safe_index)

                df_op = pd.read_csv(op_csv)
                if df_op.empty:
                    print(f"[SKIP] Empty breakdown CSV: {op_csv}")
                    continue

                op_dfs_unfiltered[op_index_name] = df_op.copy()

                if skip_cases:
                    df_op = df_op[~df_op["case"].isin(skip_cases)]
                if df_op.empty:
                    continue

                print(f"  Breakdown index: {op_index_name}  ({op_fname})")
                plot_operator_breakdown(
                    df_op, op_index_name, args.sf, args.system, out_base,
                    stats=breakdown_stats,
                )

            # Summary operator breakdown (pre/post grouping)
            if op_dfs_unfiltered:
                print(f"\n{'='*60}")
                print("Generating summary pre/post operator breakdown (SO)...")
                print(f"{'='*60}")
                for bstat in breakdown_stats:
                    plot_summary_prepost_operator(
                        op_dfs_unfiltered, args.sf, args.system, out_base,
                        stat=bstat,
                    )

    print("\nAll plots done.")


if __name__ == "__main__":
    main()
