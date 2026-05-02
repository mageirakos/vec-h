# plot_paper.py — Paper-quality plotting script derived from plot_caliper.py
# Generates:
#   - Plot 1 summary grid (detailed: Rel. Operators / Vector Search / Data Movement / Index Movement / Other)
#   - Plot 2 summary grid (full operator breakdown, coarse=False)
# Two modes:
#   - Per-index: 1x8 or 2x4 grid for a single index
#   - Multi-index: all APPROVED_INDEXES stacked as rows in a single figure
#
# Usage:
#   python plot_paper.py --sf 0.01 --benchmark vsds --system dgx-spark-02
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import math
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ==============================================================================
# 1. CONSTANTS & STYLE CONFIGURATION
# ==============================================================================

CASE_RENAME_MAPPING = {
    "5: PGV-CPU-CPU": "cpu-pgv",  # pgvector all-CPU; only present in _pg variant; placed first so the bar renders before cpu
    "0: CPU-CPU-CPU": "cpu",
    "1: GPU-CPU-CPU": "mixed",  # host + index on host
    "2: GPU-CPU-GPU": "mixed (Host D)",
    "3: GPU-GPU-GPU": "gpu",
    "4: MIXED(VS=CPU)": "mixed(VS=CPU)",
}

# pg variant: maps Maximus index name -> pgvector parsed CSV stem (without _sf_X_k_Y.csv)
PG_INDEX_MAPPING = {
    "Flat":                        "enn",
    "Cagra(C+H),64,32,NN_DESCENT": "HNSW32",
    "IVF1024(H),Flat":             "IVF1024",
}
PG_CASE_LABEL = "5: PGV-CPU-CPU"

REGIONS = ["Operators", "Data Transfers", "Other"]
RENAME_MAPPING = {
    "Operators": "Operator Execution",
    "Data Transfers": "Data Movement",
    "Other": "Other"
}

SKIP_QUERY = ["q1_start"]
# SKIP_QUERY = []

QUERY_RENAME_MAPPING = {
    "q1_start"    : "Q1",
    "q2_start"    : "Q2",
    "q10_mid"     : "Q10",
    "q11_end"     : "Q11",
    "q13_mid"     : "Q13",
    "q15_end"     : "Q15",
    "q16_start"   : "Q16",
    "q18_mid"     : "Q18",
    "q19_start"   : "Q19",
}

OPERATORS_TO_PLOT = [
    "VectorSearch", "Filter", "Project", "Join", "GroupBy",
    "OrderBy", "Limit",
    "Other"
]

# Operators folded into "Other" for cleaner legends.
# "CTE Scan" (pgvector-only) is rolled into "Operators" by vsds_benchmark so
# plot 1's Rel. Operators segment includes it; here we fold it into Other for
# plot 2's operator breakdown so a pg-only bucket doesn't clutter the legend.
_FOLD_INTO_OTHER = ["Take", "Scatter", "Gather", "CTE Scan"]

# Approved indexes for multi-index summary figures
APPROVED_INDEXES = ["Flat", "IVF1024_Flat", "IVF4096_Flat", "IVF1024(H)_Flat", "IVF4096(H)_Flat", "Cagra_64_32_NN_DESCENT", "Cagra(C+H)_64_32_NN_DESCENT"]

# Display name overrides for indexes — if an index key is present, its value is used in titles/labels
# Keys use underscores; _get_index_display_name() normalizes commas/spaces before lookup.
INDEX_DISPLAY_NAMES = {
    "Flat":                        "Exhaustive",
    "IVF1024_Flat":                "IVF1024",
    "IVF4096_Flat":                "IVF4096",
    "IVF1024(H)_Flat":             "IVF1024(H)",
    "IVF4096(H)_Flat":             "IVF4096(H)",
    "Cagra_64_32_NN_DESCENT":      "CAGRA",
    "Cagra(C+H)_64_32_NN_DESCENT": "CAGRA(C+H)",
}

def _get_index_display_name(index_desc):
    """Lookup display name, normalizing commas/spaces to underscores."""
    key = index_desc.replace(",", "_").replace(" ", "_")
    return INDEX_DISPLAY_NAMES.get(key, index_desc)

# Style tokens — sized for LaTeX papers (single-col ~3.5", double-col ~7.0")
BAR_WIDTH = 0.6
FIGSIZE_SUMMARY_WIDTH = 7.0
DPI = 600

######### 2-4
# X_ROTATION=0
# USE_ROT=False
# SUMMARY_GRID_ROWS = 2                  # fixed 2-row x 4-col layout for 8-query VSDS grids
# SUMMARY_GRID_COLS = 4
# FIGSIZE_SUMMARY_HEIGHT_PER_ROW = 2.25  # per-row height for summary grids

# TITLE_FONTSIZE   = 8
# LABEL_FONTSIZE   = 8
# XTICK_FONTSIZE   = 8
# YTICK_FONTSIZE   = 6
# VALUE_FONTSIZE   = 8
# LEGEND_FONTSIZE  = 8

######### 1-8
X_ROTATION = 30
USE_ROT = True
SUMMARY_GRID_ROWS = 1                  # fixed 1-row x 8-col layout for 8-query VSDS grids
SUMMARY_GRID_COLS = 8
FIGSIZE_SUMMARY_HEIGHT_PER_ROW = 1.0  # per-row height for summary grids

TITLE_FONTSIZE  = 7
LABEL_FONTSIZE  = 7
XTICK_FONTSIZE  = 6
YTICK_FONTSIZE  = 5
VALUE_FONTSIZE  = 6
LEGEND_FONTSIZE = 7

DEFAULT_FONT_WEIGHT = 'normal'
NORMAL_FONT_WEIGHT  = 'normal'
EDGE_COLOR = 'black'

COLOR_MAPPING = {
    # Plot 1 Categories
    "Operator Execution": "#34495e",  # Dark Blue/Gray
    "Data Movement":      "#2ecc71",  # Green
    "Other":              "#9b59b6",  # Purple

    # Detailed Plot 1 Categories
    "Rel. Operators": "#3498db",  # Blue
    "Vector Search":  "#e67e22",  # Orange
    "Index Movement": "#e74c3c",  # Red/Coastal

    # Plot 2 Operators
    "VectorSearch": "#e67e22",  # Orange
    "Filter":       "#5dade2",  # Sky Blue
    "Project":      "#f1c40f",  # Yellow
    "Join":         "#ff69b4",  # Pink
    "GroupBy":      "#34495e",  # Dark Navy
    "OrderBy":      "#c0392b",  # Dark Red
    "Limit":        "#2ecc71",  # Green
    "Take":    "#7f8c8d",  # Dark Gray
    "Scatter": "#bdc3c7",  # Light Gray
    "Gather":  "#a29bfe",  # Lavender
}

DEFAULT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Plot options
SHOW_VALUES   = True
SHOW_STD_BARS = True

# Set by main() from CLI args
_K_LABEL    = ""
_EXT        = ".jpeg"
_K_VALUE    = "100"
SHOW_TITLE  = False
_SKIP_CASES = set()

# ==============================================================================
# 2. CORE UTILITIES
# ==============================================================================

def load_data(csv_file):
    """Reads the CSV with multi-index columns."""
    df = pd.read_csv(csv_file, header=[0, 1], index_col=0)
    return df

def get_plot_path(out_dir, benchmark, sf, system, index_desc, plot_type, metric, filename, per_query=False):
    """
    Generates the hierarchy: /plots_paper/<benchmark>/sf_<sf>/<system>/<index>/<plot_type>/<metric>/<filename>
    The filename is automatically prefixed with '{system}_{benchmark}_sf{sf}_{safe_index}_{metric}_'
    so each file is self-describing without needing its directory path.
    """
    safe_idx = index_desc.replace(",", "_").replace(" ", "_")
    file_prefix = f"{system}_{benchmark}_sf{sf}_k{_K_VALUE}_{safe_idx}_{metric}_"
    prefixed_filename = file_prefix + filename
    plot_dir = os.path.join(out_dir, benchmark, f"sf_{sf}", system, index_desc, plot_type, metric)
    if per_query:
        plot_dir = os.path.join(plot_dir, "per_query")
    os.makedirs(plot_dir, exist_ok=True)
    return os.path.join(plot_dir, prefixed_filename)

# ==============================================================================
# 3. DATA PREPARATION LOGIC
# ==============================================================================

def _is_enn_index(index_desc):
    """ENN (exhaustive nearest neighbor) is implemented with the Flat index.
    Display name 'Exhaustive' is the canonical marker."""
    if not index_desc:
        return False
    return _get_index_display_name(index_desc) == "Exhaustive"


def get_query_data(df, row_idx, detailed=False, index_desc=None):
    """Unified data view extractor for Plot 1 variants."""
    row = df.loc[row_idx]
    df_plot = row.unstack(level=0)
    # Reorder rows to follow CASE_RENAME_MAPPING insertion order so bars render in
    # the desired sequence (pg first, then cpu, mixed, ...). Cases not present are dropped.
    ordered_cases = [c for c in CASE_RENAME_MAPPING.keys() if c in df_plot.index]
    df_plot = df_plot.loc[ordered_cases]
    df_plot = df_plot[~df_plot.index.isin(_SKIP_CASES)]
    df_plot = df_plot.rename(index=CASE_RENAME_MAPPING)
    missing_case_mask = df_plot.isna().all(axis=1)

    def get_col(name):
        if name in df_plot.columns:
            return df_plot[name]
        return pd.Series(np.nan, index=df_plot.index)

    if not detailed:
        cols = [r for r in REGIONS if r in df_plot.columns]
        plot_data = df_plot[cols].copy().rename(columns=RENAME_MAPPING)
        plot_data = plot_data.fillna(0.0)
        plot_data.loc[missing_case_mask, :] = np.nan
    else:
        ops, vs = get_col('Operators').fillna(0.0), get_col('VectorSearch').fillna(0.0)
        dt, im  = get_col('Data Transfers').fillna(0.0), get_col('IndexMovement').fillna(0.0)
        oth     = get_col('Other').fillna(0.0)

        df_plot['Rel. Operators'] = (ops - vs).clip(lower=0)
        df_plot['Vector Search']  = vs

        if _is_enn_index(index_desc):
            # ENN simplification: ENN runs against the Faiss-managed Flat
            # index because the reviews dataset's `rv_embedding` column is an
            # Arrow `large_list<float32>` that cuDF cannot lift to GPU
            # directly (int32 indexing limit on lists). The Flat index is
            # therefore the carrier of the embedding data, and what the
            # parser labels "Index Movement" is in fact the same data
            # movement as moving the embedding column to GPU. We fold it
            # into "Data Movement" so the ENN bars don't visually separate
            # data and index movement that are physically the same bytes.
            df_plot['Data Movement']  = dt
            df_plot['Other']          = oth
            detailed_cols = ["Rel. Operators", "Vector Search", "Data Movement", "Other"]
        else:
            df_plot['Data Movement']  = (dt - im).clip(lower=0)
            df_plot['Index Movement'] = im
            df_plot['Other']          = oth
            detailed_cols = ["Rel. Operators", "Vector Search", "Data Movement", "Index Movement", "Other"]

        plot_data = df_plot[detailed_cols].copy()
        plot_data.loc[missing_case_mask, :] = np.nan

    return plot_data

def get_query_operator_data(df, row_idx, coarse=False):
    """Extract operator breakdown for a single query across all execution cases."""
    row_data = df.loc[row_idx]
    cases = [c for c in CASE_RENAME_MAPPING.keys() if c not in _SKIP_CASES]
    plot_rows = []

    for case_label in cases:
        if not coarse:
            data = {op: row_data.get((op, case_label), np.nan) for op in OPERATORS_TO_PLOT}
            # Fold Take/Scatter/Gather into Other
            fold_vals = [row_data.get((op, case_label), np.nan) for op in _FOLD_INTO_OTHER]
            fold_sum = float(np.nansum(fold_vals))
            if "Other" in data:
                other_val = data["Other"]
                data["Other"] = (0.0 if pd.isna(other_val) else other_val) + fold_sum
            has_any = any(pd.notna(v) for v in data.values())
            if has_any:
                data = {k: (0.0 if pd.isna(v) else v) for k, v in data.items()}
        else:
            vs_raw  = row_data.get(("VectorSearch", case_label), np.nan)
            dc_raw  = row_data.get(("Other", case_label), np.nan)
            rel_ops_raw = [row_data.get((op, case_label), np.nan)
                           for op in OPERATORS_TO_PLOT if op not in ["VectorSearch", "Other"]]
            if pd.isna(vs_raw) and pd.isna(dc_raw) and all(pd.isna(v) for v in rel_ops_raw):
                data = {"Vector Search": np.nan, "Rel. Operators": np.nan, "Other": np.nan}
            else:
                vs_val = 0.0 if pd.isna(vs_raw) else vs_raw
                dc_val = 0.0 if pd.isna(dc_raw) else dc_raw
                rel_val = float(np.nansum(rel_ops_raw))
                data = {"Vector Search": vs_val, "Rel. Operators": rel_val, "Other": dc_val}
        plot_rows.append(data)

    return pd.DataFrame(plot_rows, index=[CASE_RENAME_MAPPING[c] for c in cases])


def annotate_missing_cases(ax, plot_data):
    """Annotate x positions where all stacked values are missing or zero."""
    
    missing_mask = plot_data.isna().all(axis=1)
    zero_mask = plot_data.fillna(0.0).eq(0.0).all(axis=1)
    na_mask = missing_mask | zero_mask
    if not na_mask.any():
        return
    y_min, y_max = ax.get_ylim()
    y_pos = y_min + (y_max - y_min) * 0.03
    for i, is_na in enumerate(na_mask.values):
        if is_na:
            ax.text(i, y_pos, "N/A", ha='center', va='bottom',
                    fontweight=DEFAULT_FONT_WEIGHT, fontsize=VALUE_FONTSIZE)

# ==============================================================================
# 4. PLOT 1 SUMMARY GRID (per-index)
# ==============================================================================

def plot_1_summary_grid(df, target_indices, metric_suffix, index_desc, sf, out_dir, metric, system, benchmark, detailed=False, fixed_ylim=None):
    """Generates grid summaries for Plot 1."""
    if not target_indices: return []
    num_q = len(target_indices)
    rows, cols = SUMMARY_GRID_ROWS, SUMMARY_GRID_COLS

    if detailed and _is_enn_index(index_desc):
        print(f"  [PLOT1] ENN index '{index_desc}': folding 'Index Movement' into "
              f"'Data Movement' (Flat index carries the reviews `rv_embedding` "
              f"large_list column to work around cuDF int32 list limit; index "
              f"and data movement are physically the same bytes).")

    fig, axes = plt.subplots(rows, cols, figsize=(FIGSIZE_SUMMARY_WIDTH, FIGSIZE_SUMMARY_HEIGHT_PER_ROW * rows))
    axes = np.array(axes).reshape(-1) if num_q > 1 else [axes]

    for i, row_idx in enumerate(target_indices):
        q = str(row_idx)[:-len(metric_suffix)]
        plot_data = get_query_data(df, row_idx, detailed=detailed, index_desc=index_desc)
        ax = axes[i]

        plot_colors = [COLOR_MAPPING.get(col, DEFAULT_COLORS[j % len(DEFAULT_COLORS)])
                       for j, col in enumerate(plot_data.columns)]

        plot_data.plot(kind='bar', stacked=True, ax=ax, edgecolor=EDGE_COLOR,
                       width=BAR_WIDTH, logy=False, color=plot_colors)
        ax.set_title(QUERY_RENAME_MAPPING.get(q, q), fontsize=TITLE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
        if i % SUMMARY_GRID_COLS == 0:
            ax.set_ylabel("Runtime [ms]", fontweight=DEFAULT_FONT_WEIGHT, fontsize=LABEL_FONTSIZE)
        if not USE_ROT:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=X_ROTATION,
                               fontsize=XTICK_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=X_ROTATION, ha='right',
                               fontsize=XTICK_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
        ax.tick_params(axis='x', pad=0)
        ax.tick_params(axis='y', labelsize=YTICK_FONTSIZE)
        if ax.get_legend(): ax.get_legend().remove()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#888888')
        ax.spines['bottom'].set_color('#888888')

        err_arr = None
        if metric in ["mean"] and SHOW_STD_BARS:
            std_row_idx = f"{q}_std"
            if std_row_idx in df.index:
                std_df = df.loc[std_row_idx].unstack(level=0).rename(index=CASE_RENAME_MAPPING)
                if 'Total' in std_df.columns:
                    errs = std_df['Total'].reindex(plot_data.index).fillna(0)
                    totals = plot_data.sum(axis=1)
                    x_coords = np.arange(len(plot_data))
                    ax.errorbar(x_coords, totals, yerr=errs, fmt='none', ecolor='black',
                                capsize=6, capthick=2, elinewidth=2, zorder=10)
                    err_arr = errs.values

        if fixed_ylim:
            ax.set_ylim(fixed_ylim)
        elif SHOW_VALUES:
            totals = plot_data.sum(axis=1)
            curr_min, curr_max = ax.get_ylim()
            max_y = curr_max
            if err_arr is not None:
                max_y_with_err = np.max(totals.values + err_arr)
                max_y = max(max_y, max_y_with_err)
            ax.set_ylim(bottom=curr_min, top=max_y * 1.10)

        if SHOW_VALUES:
            totals = plot_data.sum(axis=1)
            curr_min, curr_max = ax.get_ylim()
            pad = (curr_max - curr_min) * 0.01
            for j, val in enumerate(totals):
                if not np.isnan(val) and val > 0:
                    y_pos = val
                    if err_arr is not None and j < len(err_arr):
                        y_pos += err_arr[j]
                    ax.text(j, y_pos + pad, f"{val:.0f}", ha='center', va='bottom',
                            fontweight=DEFAULT_FONT_WEIGHT, fontsize=VALUE_FONTSIZE)

        annotate_missing_cases(ax, plot_data)

    for i in range(num_q, len(axes)): fig.delaxes(axes[i])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False,
               prop={'weight': NORMAL_FONT_WEIGHT, 'size': LEGEND_FONTSIZE})

    if SHOW_TITLE:
        tag = "Detailed " if detailed else ""
        title = f"{tag}Summary Breakdown ({metric.upper()}) - {system} ({index_desc}, SF={sf}{_K_LABEL})"
        plt.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)

    fig.tight_layout(pad=0, w_pad=0.2, rect=[0, 0.16, 1, 1.0])

    prefix = "detailed_" if detailed else ""
    filename = f"summary_highlevel_breakdown{_EXT}"
    out_path = get_plot_path(out_dir, benchmark, sf, system, index_desc, "plot_1", metric, filename)
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return [out_path]

# ==============================================================================
# 5. PLOT 2 SUMMARY GRID (per-index)
# ==============================================================================

def plot_2_summary_grid(df, target_indices, metric_suffix, index_desc, sf, out_dir, metric, system, benchmark, coarse=False):
    """Generates grid summaries for Plot 2 (one subplot per query, operator breakdown across cases)."""
    if not target_indices: return []
    num_q = len(target_indices)
    rows, cols = SUMMARY_GRID_ROWS, SUMMARY_GRID_COLS

    fig, axes = plt.subplots(rows, cols, figsize=(FIGSIZE_SUMMARY_WIDTH, FIGSIZE_SUMMARY_HEIGHT_PER_ROW * rows))
    axes = np.array(axes).reshape(-1) if num_q > 1 else [axes]

    for i, row_idx in enumerate(target_indices):
        q = str(row_idx)[:-len(metric_suffix)]
        plot_data = get_query_operator_data(df, row_idx, coarse=coarse)
        ax = axes[i]

        plot_colors = [COLOR_MAPPING.get(col, DEFAULT_COLORS[j % len(DEFAULT_COLORS)])
                       for j, col in enumerate(plot_data.columns)]

        plot_data.plot(kind='bar', stacked=True, ax=ax, edgecolor=EDGE_COLOR,
                       width=BAR_WIDTH, logy=False, color=plot_colors)
        ax.set_title(QUERY_RENAME_MAPPING.get(q, q), fontsize=TITLE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
        if i % SUMMARY_GRID_COLS == 0:
            ax.set_ylabel("Runtime [ms]", fontweight=DEFAULT_FONT_WEIGHT, fontsize=LABEL_FONTSIZE)
        if not USE_ROT:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=X_ROTATION,
                               fontsize=XTICK_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=X_ROTATION, ha='right',
                               fontsize=XTICK_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
        ax.tick_params(axis='x', pad=0)
        ax.tick_params(axis='y', labelsize=YTICK_FONTSIZE)
        if ax.get_legend(): ax.get_legend().remove()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#888888')
        ax.spines['bottom'].set_color('#888888')

        err_arr = None
        if metric in ["mean"] and SHOW_STD_BARS:
            std_row_idx = f"{q}_std"
            if std_row_idx in df.index:
                std_df = df.loc[std_row_idx].unstack(level=0)
                if 'Operators' in std_df.columns:
                    cases = list(CASE_RENAME_MAPPING.keys())
                    errs = std_df['Operators'].reindex(cases).fillna(0)
                    errs.index = [CASE_RENAME_MAPPING[c] for c in cases]
                    errs = errs.reindex(plot_data.index).fillna(0)
                    totals = plot_data.sum(axis=1)
                    x_coords = np.arange(len(plot_data))
                    ax.errorbar(x_coords, totals, yerr=errs, fmt='none', ecolor='black',
                                capsize=6, capthick=2, elinewidth=2, zorder=10)
                    err_arr = errs.values

        if SHOW_VALUES:
            totals = plot_data.sum(axis=1)
            curr_min, curr_max = ax.get_ylim()
            max_y = curr_max
            if err_arr is not None:
                max_y_with_err = np.max(totals.values + err_arr)
                max_y = max(max_y, max_y_with_err)
            ax.set_ylim(bottom=curr_min, top=max_y * 1.10)
            pad = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
            for j, val in enumerate(totals):
                if not np.isnan(val) and val > 0:
                    y_pos = val
                    if err_arr is not None and j < len(err_arr):
                        y_pos += err_arr[j]
                    ax.text(j, y_pos + pad, f"{val:.0f}", ha='center', va='bottom',
                            fontweight=DEFAULT_FONT_WEIGHT, fontsize=VALUE_FONTSIZE)

        annotate_missing_cases(ax, plot_data)

    for i in range(num_q, len(axes)): fig.delaxes(axes[i])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False,
               prop={'weight': NORMAL_FONT_WEIGHT, 'size': LEGEND_FONTSIZE})

    if SHOW_TITLE:
        tag = "Coarse " if coarse else ""
        title = f"{tag}Operator Breakdown Summary ({metric.upper()}) - {system} ({index_desc}, SF={sf}{_K_LABEL})"
        plt.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)

    fig.tight_layout(pad=0, w_pad=0.2, rect=[0, 0.12, 1, 1.0])

    filename = f"operator_{'coarse_' if coarse else ''}breakdown_grid{_EXT}"
    out_path = get_plot_path(out_dir, benchmark, sf, system, index_desc, "plot_2", metric, filename)
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return [out_path]

# ==============================================================================
# 6. MULTI-INDEX PLOTS
# ==============================================================================

def plot_1_multi_index_summary(index_entries, sf, out_dir, metric, system, benchmark, suffix=""):
    """
    Generates a multi-index Plot 1 summary (detailed breakdown).
    index_entries: list of (df, index_desc, targets, metric_suffix) tuples — one per approved index.
    Layout: (n_indexes * SUMMARY_GRID_ROWS) rows x SUMMARY_GRID_COLS cols.
    Subplot titles: "{index_label}: {query_name}".
    Y-label on leftmost subplot of each index block's first row only.
    Single shared legend at figure bottom.
    suffix: appended to output filename (e.g. "_no_cpu").
    """
    if not index_entries: return []

    n_indexes  = len(index_entries)
    total_rows = n_indexes * SUMMARY_GRID_ROWS
    cols       = SUMMARY_GRID_COLS

    fig, axes = plt.subplots(
        total_rows, cols,
        figsize=(FIGSIZE_SUMMARY_WIDTH, FIGSIZE_SUMMARY_HEIGHT_PER_ROW * total_rows)
    )
    axes = np.array(axes).reshape(total_rows, cols)

    all_handles, all_labels = None, None

    for bi, (df, index_desc, targets, metric_suffix) in enumerate(index_entries):
        index_label = _get_index_display_name(index_desc)

        if _is_enn_index(index_desc):
            print(f"  [MULTI-IDX] ENN index '{index_desc}': folding 'Index Movement' "
                  f"into 'Data Movement' (Flat index carries the reviews "
                  f"`rv_embedding` large_list column to work around cuDF int32 "
                  f"list limit; index and data movement are physically the same bytes).")

        for qi, row_idx in enumerate(targets):
            q = str(row_idx)[:-len(metric_suffix)]
            local_row, local_col = divmod(qi, cols)
            global_row = bi * SUMMARY_GRID_ROWS + local_row
            ax = axes[global_row, local_col]

            plot_data = get_query_data(df, row_idx, detailed=True, index_desc=index_desc)

            plot_colors = [COLOR_MAPPING.get(col, DEFAULT_COLORS[j % len(DEFAULT_COLORS)])
                           for j, col in enumerate(plot_data.columns)]

            plot_data.plot(kind='bar', stacked=True, ax=ax, edgecolor=EDGE_COLOR,
                           width=BAR_WIDTH, logy=False, color=plot_colors)

            query_name = QUERY_RENAME_MAPPING.get(q, q)
            if bi == 0:
                ax.set_title(query_name, fontsize=TITLE_FONTSIZE,
                             fontweight=DEFAULT_FONT_WEIGHT)

            # Row label: index name on leftmost subplot of each row
            if local_col == 0:
                ax.set_ylabel(index_label, fontsize=LABEL_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
            else:
                ax.set_ylabel("")

            # X-tick labels only on the bottom row of the multi-index grid
            is_bottom_row = (bi == n_indexes - 1)
            if is_bottom_row:
                if not USE_ROT:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=X_ROTATION,
                                       fontsize=XTICK_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
                else:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=X_ROTATION, ha='right',
                                       fontsize=XTICK_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis='x', pad=0)
            ax.tick_params(axis='y', labelsize=YTICK_FONTSIZE)
            if ax.get_legend(): ax.get_legend().remove()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#888888')
            ax.spines['bottom'].set_color('#888888')

            err_arr = None
            if metric in ["mean"] and SHOW_STD_BARS:
                std_row_idx = f"{q}_std"
                if std_row_idx in df.index:
                    std_df = df.loc[std_row_idx].unstack(level=0).rename(index=CASE_RENAME_MAPPING)
                    if 'Total' in std_df.columns:
                        errs = std_df['Total'].reindex(plot_data.index).fillna(0)
                        totals = plot_data.sum(axis=1)
                        x_coords = np.arange(len(plot_data))
                        ax.errorbar(x_coords, totals, yerr=errs, fmt='none', ecolor='black',
                                    capsize=6, capthick=2, elinewidth=2, zorder=10)
                        err_arr = errs.values

            if SHOW_VALUES:
                totals = plot_data.sum(axis=1)
                curr_min, curr_max = ax.get_ylim()
                max_y = curr_max
                if err_arr is not None:
                    max_y_with_err = np.max(totals.values + err_arr)
                    max_y = max(max_y, max_y_with_err)
                ax.set_ylim(bottom=curr_min, top=max_y * 1.10)
                pad = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
                for j, val in enumerate(totals):
                    if not np.isnan(val) and val > 0:
                        y_pos = val
                        if err_arr is not None and j < len(err_arr):
                            y_pos += err_arr[j]
                        ax.text(j, y_pos + pad, f"{val:.0f}", ha='center', va='bottom',
                                fontweight=DEFAULT_FONT_WEIGHT, fontsize=VALUE_FONTSIZE)

            annotate_missing_cases(ax, plot_data)

            # Capture handles from the subplot with the *most* categories so the
            # shared legend covers every bar segment used anywhere in the figure.
            # ENN rows produce only 4 categories (Index Movement folded into
            # Data Movement); ANN rows produce 5. Without this, the first row
            # processed wins and the legend may be missing entries.
            handles, labels = ax.get_legend_handles_labels()
            if all_handles is None or len(labels) > len(all_labels):
                all_handles, all_labels = handles, labels

        # Hide unused subplots in this index block
        for qi in range(len(targets), SUMMARY_GRID_ROWS * cols):
            local_row, local_col = divmod(qi, cols)
            global_row = bi * SUMMARY_GRID_ROWS + local_row
            fig.delaxes(axes[global_row, local_col])

    if all_handles:
        fig.legend(all_handles, all_labels, loc='upper center', ncol=len(all_labels),
                   frameon=False, prop={'weight': NORMAL_FONT_WEIGHT, 'size': LEGEND_FONTSIZE})

    fig.text(0.00, 0.5, "Runtime [ms]", va='center', ha='center',
             rotation='vertical', fontsize=LABEL_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
    fig.tight_layout(pad=0, w_pad=0.2, rect=[0.02, 0, 1, 0.92])

    multi_dir = os.path.join(out_dir, benchmark, f"sf_{sf}", system, "multi_index", "plot_1", metric)
    os.makedirs(multi_dir, exist_ok=True)
    out_path = os.path.join(multi_dir, f"multi_index_summary_detailed{suffix}{_EXT}")
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path}")
    return [out_path]


def plot_2_multi_index_summary(index_entries, sf, out_dir, metric, system, benchmark, suffix=""):
    """
    Generates a multi-index Plot 2 summary (full operator breakdown, coarse=False).
    index_entries: list of (df, index_desc, targets, metric_suffix) tuples — one per approved index.
    Layout: (n_indexes * SUMMARY_GRID_ROWS) rows x SUMMARY_GRID_COLS cols.
    Subplot titles: "{index_label}: {query_name}".
    Y-label on leftmost subplot of each index block's first row only.
    Single shared legend at figure bottom.
    suffix: appended to output filename (e.g. "_no_cpu").
    """
    if not index_entries: return []

    n_indexes  = len(index_entries)
    total_rows = n_indexes * SUMMARY_GRID_ROWS
    cols       = SUMMARY_GRID_COLS

    fig, axes = plt.subplots(
        total_rows, cols,
        figsize=(FIGSIZE_SUMMARY_WIDTH, FIGSIZE_SUMMARY_HEIGHT_PER_ROW * total_rows)
    )
    axes = np.array(axes).reshape(total_rows, cols)

    all_handles, all_labels = None, None

    for bi, (df, index_desc, targets, metric_suffix) in enumerate(index_entries):
        index_label = _get_index_display_name(index_desc)

        for qi, row_idx in enumerate(targets):
            q = str(row_idx)[:-len(metric_suffix)]
            local_row, local_col = divmod(qi, cols)
            global_row = bi * SUMMARY_GRID_ROWS + local_row
            ax = axes[global_row, local_col]

            plot_data = get_query_operator_data(df, row_idx, coarse=False)

            plot_colors = [COLOR_MAPPING.get(col, DEFAULT_COLORS[j % len(DEFAULT_COLORS)])
                           for j, col in enumerate(plot_data.columns)]

            plot_data.plot(kind='bar', stacked=True, ax=ax, edgecolor=EDGE_COLOR,
                           width=BAR_WIDTH, logy=False, color=plot_colors)

            query_name = QUERY_RENAME_MAPPING.get(q, q)
            if bi == 0:
                ax.set_title(query_name, fontsize=TITLE_FONTSIZE,
                             fontweight=DEFAULT_FONT_WEIGHT)

            # Row label: index name on leftmost subplot of each row
            if local_col == 0:
                ax.set_ylabel(index_label, fontsize=LABEL_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
            else:
                ax.set_ylabel("")

            # X-tick labels only on the bottom row of the multi-index grid
            is_bottom_row = (bi == n_indexes - 1)
            if is_bottom_row:
                if not USE_ROT:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=X_ROTATION,
                                       fontsize=XTICK_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
                else:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=X_ROTATION, ha='right',
                                       fontsize=XTICK_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis='x', pad=0)
            ax.tick_params(axis='y', labelsize=YTICK_FONTSIZE)
            if ax.get_legend(): ax.get_legend().remove()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#888888')
            ax.spines['bottom'].set_color('#888888')

            err_arr = None
            if metric in ["mean"] and SHOW_STD_BARS:
                std_row_idx = f"{q}_std"
                if std_row_idx in df.index:
                    std_df = df.loc[std_row_idx].unstack(level=0)
                    if 'Operators' in std_df.columns:
                        cases = list(CASE_RENAME_MAPPING.keys())
                        errs = std_df['Operators'].reindex(cases).fillna(0)
                        errs.index = [CASE_RENAME_MAPPING[c] for c in cases]
                        errs = errs.reindex(plot_data.index).fillna(0)
                        totals = plot_data.sum(axis=1)
                        x_coords = np.arange(len(plot_data))
                        ax.errorbar(x_coords, totals, yerr=errs, fmt='none', ecolor='black',
                                    capsize=6, capthick=2, elinewidth=2, zorder=10)
                        err_arr = errs.values

            if SHOW_VALUES:
                totals = plot_data.sum(axis=1)
                curr_min, curr_max = ax.get_ylim()
                max_y = curr_max
                if err_arr is not None:
                    max_y_with_err = np.max(totals.values + err_arr)
                    max_y = max(max_y, max_y_with_err)
                ax.set_ylim(bottom=curr_min, top=max_y * 1.10)
                pad = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
                for j, val in enumerate(totals):
                    if not np.isnan(val) and val > 0:
                        y_pos = val
                        if err_arr is not None and j < len(err_arr):
                            y_pos += err_arr[j]
                        ax.text(j, y_pos + pad, f"{val:.0f}", ha='center', va='bottom',
                                fontweight=DEFAULT_FONT_WEIGHT, fontsize=VALUE_FONTSIZE)

            annotate_missing_cases(ax, plot_data)

            if all_handles is None:
                all_handles, all_labels = ax.get_legend_handles_labels()

        for qi in range(len(targets), SUMMARY_GRID_ROWS * cols):
            local_row, local_col = divmod(qi, cols)
            global_row = bi * SUMMARY_GRID_ROWS + local_row
            fig.delaxes(axes[global_row, local_col])

    if all_handles:
        fig.legend(all_handles, all_labels, loc='upper center',
                   ncol=len(all_labels), frameon=False,
                   prop={'weight': NORMAL_FONT_WEIGHT, 'size': LEGEND_FONTSIZE})

    fig.text(0.00, 0.5, "Runtime [ms]", va='center', ha='center',
             rotation='vertical', fontsize=LABEL_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
    fig.tight_layout(pad=0, w_pad=0.2, rect=[0.02, 0, 1, 0.92])

    multi_dir = os.path.join(out_dir, benchmark, f"sf_{sf}", system, "multi_index", "plot_2", metric)
    os.makedirs(multi_dir, exist_ok=True)
    out_path = os.path.join(multi_dir, f"multi_index_operator_breakdown{suffix}{_EXT}")
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path}")
    return [out_path]

# ==============================================================================
# 7. ORCHESTRATION
# ==============================================================================

def generate_plots(csv_file, index_desc, sf, out_dir, metric, system, benchmark):
    """Per-index: generate Plot 1 (detailed) and Plot 2 summary grids."""
    df = load_data(csv_file)
    metric_ext = f"_{metric}"
    targets = [idx for idx in df.index
               if str(idx).endswith(metric_ext) and str(idx)[:-len(metric_ext)] not in SKIP_QUERY]

    all_p = []
    all_p += plot_1_summary_grid(df, targets, metric_ext, index_desc, sf, out_dir, metric,
                                 system, benchmark, detailed=True)
    all_p += plot_2_summary_grid(df, targets, metric_ext, index_desc, sf, out_dir, metric,
                                 system, benchmark, coarse=False)
    return all_p


def _merge_pg_into_maximus_df(max_df, pg_csv_path):
    """Append a synthetic case column (PG_CASE_LABEL) to a Maximus df, sourced from
    a pgvector CSV's '0: CPU-CPU-CPU' values. Rows missing in pg become NaN
    (rendered as N/A by annotate_missing_cases). Returns a new df; original untouched.
    """
    pg_df = load_data(pg_csv_path)
    out = max_df.copy()
    metrics = max_df.columns.get_level_values(0).unique()
    pg_src_case = "0: CPU-CPU-CPU"
    for metric in metrics:
        if (metric, pg_src_case) in pg_df.columns:
            src = pg_df[(metric, pg_src_case)]
        else:
            src = pd.Series(np.nan, index=pg_df.index)
        out[(metric, PG_CASE_LABEL)] = src.reindex(out.index)
    out = out.sort_index(axis=1)
    return out


def _build_pg_index_entries(index_entries, pg_dir, benchmark, sf, k):
    """Filter index_entries to PG_INDEX_MAPPING keys and merge pg case into each df.
    Missing pg CSVs result in NaN bars (still kept; annotated N/A).
    """
    pg_subdir = os.path.join(pg_dir, "parse_postgres", benchmark, "scaled")
    pg_entries = []
    for df, norm_idx, targets, metric_suffix in index_entries:
        if norm_idx not in PG_INDEX_MAPPING:
            continue
        pg_stem = PG_INDEX_MAPPING[norm_idx]
        pg_csv = os.path.join(pg_subdir, f"pgvector_{benchmark}_{pg_stem}_sf_{sf}_k_{k}.csv")
        if os.path.exists(pg_csv):
            print(f"  [_pg] Merging pg CSV for index '{norm_idx}': {pg_csv}")
            merged_df = _merge_pg_into_maximus_df(df, pg_csv)
        else:
            print(f"  [_pg] No pg CSV for index '{norm_idx}' (looked for {pg_csv}); pg bar will be N/A.")
            merged_df = df.copy()
            metrics = merged_df.columns.get_level_values(0).unique()
            for m in metrics:
                merged_df[(m, PG_CASE_LABEL)] = np.nan
            merged_df = merged_df.sort_index(axis=1)
        pg_entries.append((merged_df, norm_idx, targets, metric_suffix))
    return pg_entries


def _filter_case4_index_entries(index_entries, case4_label="4: MIXED(VS=CPU)"):
    """Keep only indexes whose df has at least one non-NaN value in the Case 4 column.
    Prevents mixed 3-bar/4-bar rendering within the _case4 variant."""
    filtered = []
    for entry in index_entries:
        df = entry[0]
        metrics = df.columns.get_level_values(0).unique()
        has_case4 = any(
            (m, case4_label) in df.columns and df[(m, case4_label)].notna().any()
            for m in metrics
        )
        if has_case4:
            filtered.append(entry)
        else:
            print(f"  [_case4] Dropping index '{entry[1]}' (no Case 4 data).")
    return filtered


def _is_h_variant(norm_idx):
    """True if norm_idx names an H/CH variant (IVF(H), Cagra(C+H))."""
    return "(H)" in norm_idx or "(C+H)" in norm_idx


def _merge_case4_from_base(h_entries, index_entries, case4_label="4: MIXED(VS=CPU)"):
    """For each H/CH entry, copy the Case 4 column from the matching base entry.
    Case 4 runs with VS=CPU, so the H/CH optimization has no effect — base data
    applies verbatim. Explicit here (vs. in parse_caliper) so the CSVs stay a
    one-to-one mirror of runs; the equivalence lives next to the plot that needs it."""
    def _base_norm(n):
        return n.replace("(C+H)", "").replace("(H)", "")
    by_norm = {ni: df for (df, ni, _, _) in index_entries}
    out = []
    for df, ni, targets, ms in h_entries:
        base_ni = _base_norm(ni)
        if ni == base_ni or base_ni not in by_norm:
            out.append((df, ni, targets, ms))
            continue
        base_df = by_norm[base_ni]
        merged = df.copy()
        for m in merged.columns.get_level_values(0).unique():
            col = (m, case4_label)
            if col in base_df.columns:
                merged[col] = base_df[col].reindex(merged.index)
        merged = merged.sort_index(axis=1)
        out.append((merged, ni, targets, ms))
    return out


def generate_multi_index_plots(index_entries, sf, out_dir, metric, system, benchmark,
                               pg_dir=None, k="100"):
    """Multi-index: all approved indexes stacked as rows in a single figure.
    Generates: full (cases 0/1/3), _no_cpu (cases 1/3), _case4 (cases 0/1/3/4 on
    indexes with hybrid data), and (if pg_dir) _pg variant comparing pgvector vs MaxVec."""
    global _SKIP_CASES
    all_p = []
    saved_skip = _SKIP_CASES

    # Default "full": cases 0/1/3. Drop case 4 (rendered separately in _case4 so
    # inconsistent bar counts across indexes don't hide it at the bottom of a subplot)
    # and the synthetic pg case (only added in _pg).
    _SKIP_CASES = saved_skip | {"4: MIXED(VS=CPU)", PG_CASE_LABEL}
    all_p += plot_1_multi_index_summary(index_entries, sf, out_dir, metric, system, benchmark)
    all_p += plot_2_multi_index_summary(index_entries, sf, out_dir, metric, system, benchmark)

    # "no cpu" variants: keep only case 1 (mixed) and case 3 (gpu).
    # Drop case 0 (CPU-CPU-CPU), case 2 (GPU-CPU-GPU), case 4 (MIXED(VS=CPU)), and pg.
    _SKIP_CASES = saved_skip | {"0: CPU-CPU-CPU", "2: GPU-CPU-GPU", "4: MIXED(VS=CPU)", PG_CASE_LABEL}
    all_p += plot_1_multi_index_summary(index_entries, sf, out_dir, metric, system, benchmark, suffix="_no_cpu")
    all_p += plot_2_multi_index_summary(index_entries, sf, out_dir, metric, system, benchmark, suffix="_no_cpu")

    # Case 4 variants:
    # - _case4:         base IVF/Cagra + Flat. Cases 0/1/3/4 from actual runs.
    # - _case4_h:       IVF(H)/Cagra(C+H) + Flat for cases 0..3, Case 4 borrowed
    #                   from the matching base CSV (H/CH no-op when VS=CPU).
    # - _case4_minimal: same as _case4 but drops Case 1 (mixed) and Case 2.
    base_case4 = _filter_case4_index_entries(
        [e for e in index_entries if not _is_h_variant(e[1])]
    )
    h_case4 = _filter_case4_index_entries(_merge_case4_from_base(
        [e for e in index_entries if _is_h_variant(e[1]) or e[1] == "Flat"],
        index_entries,
    ))

    def _emit(entries, suffix, skip):
        global _SKIP_CASES
        _SKIP_CASES = skip
        all_p.extend(plot_1_multi_index_summary(entries, sf, out_dir, metric, system, benchmark, suffix=suffix))
        all_p.extend(plot_2_multi_index_summary(entries, sf, out_dir, metric, system, benchmark, suffix=suffix))

    if base_case4:
        _emit(base_case4, "_case4", saved_skip | {PG_CASE_LABEL})
        _emit(base_case4, "_case4_minimal",
              saved_skip | {"1: GPU-CPU-CPU", "2: GPU-CPU-GPU", PG_CASE_LABEL})
    if h_case4:
        _emit(h_case4, "_case4_h", saved_skip | {PG_CASE_LABEL})
    if not (base_case4 or h_case4):
        print("  [_case4] No index has Case 4 data; skipping _case4 variants.")

    # "_pg" variant: pgvector vs MaxVec on CPU-mapped indexes only.
    # Keep cases 0/1/2/3 + pg; drop case 4 (redundant for the pg comparison).
    if pg_dir:
        pg_entries = _build_pg_index_entries(index_entries, pg_dir, benchmark, sf, k)
        if pg_entries:
            _SKIP_CASES = saved_skip | {"4: MIXED(VS=CPU)"}
            all_p += plot_1_multi_index_summary(pg_entries, sf, out_dir, metric, system, benchmark, suffix="_pg")
            all_p += plot_2_multi_index_summary(pg_entries, sf, out_dir, metric, system, benchmark, suffix="_pg")
        else:
            print("  [_pg] No matching index entries (none of Flat/Cagra/IVF1024 present); skipping _pg variant.")

    _SKIP_CASES = saved_skip
    return all_p

# ==============================================================================
# 8. MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Paper-quality VSDS results plotter")
    parser.add_argument("--in_dir",    default="./parse_caliper", help="Parsed CSV directory")
    parser.add_argument("--sf",        required=True,             help="Scale factor(s), comma-separated")
    parser.add_argument("--csv",                                  help="Specific CSV path override")
    parser.add_argument("--out_dir",   default="plots_paper",           help="Output base directory")
    parser.add_argument("--system",    default="dgx-spark-02",    help="System identifier")
    parser.add_argument("--benchmark", default="vsds",            help="Benchmark(s), comma-separated")
    parser.add_argument("--metric",    nargs="+", default=["median"],help="Metrics to process")
    parser.add_argument("--show-values", action="store_true", help="Show bar value annotations (off by default)")
    parser.add_argument("--std-bars", action="store_true", help="Show std error bars (off by default)")
    parser.add_argument("--skip-cases",   default="2",
                        help="Case numbers to skip (default: '2' = GPU-CPU-GPU). E.g. '1,2'")
    parser.add_argument("--title",        action="store_true", help="Show plot titles")
    parser.add_argument("--k",            default="100",
                        help="k value to select (default: 100). Matches files with '_k_<value>' tag.")
    parser.add_argument("--format",       default="jpeg", choices=["jpeg", "pdf", "png", "svg"],
                        help="Output image format. Use 'pdf' for vector quality in LaTeX papers.")
    parser.add_argument("--pg_dir",       default=None,
                        help="Postgres run directory containing parse_postgres/<benchmark>/scaled/. "
                             "When set, also emits a _pg variant comparing pgvector vs MaxVec on "
                             "CPU-mapped indexes (Flat/Cagra/IVF1024).")
    args = parser.parse_args()

    global SHOW_VALUES, SHOW_STD_BARS, _K_LABEL, _EXT, _K_VALUE, SHOW_TITLE, _SKIP_CASES
    SHOW_VALUES   = args.show_values
    SHOW_STD_BARS = args.std_bars
    _K_LABEL      = f", K={args.k}"
    _K_VALUE      = args.k
    _EXT          = f".{args.format}"
    SHOW_TITLE    = args.title
    _SKIP_CASES   = {f"{n}: " + {"0": "CPU-CPU-CPU", "1": "GPU-CPU-CPU",
                                  "2": "GPU-CPU-GPU", "3": "GPU-GPU-GPU"}[n]
                     for n in args.skip_cases.split(",") if n.strip() in ("0", "1", "2", "3")}

    k_suffixes = [f"_sf_{{}}_k_{args.k}.csv", "_sf_{{}}.csv"]

    all_done   = []
    benchmarks = [b.strip() for b in args.benchmark.split(",")]
    sfs        = [s.strip() for s in args.sf.split(",")]

    for benchmark in benchmarks:
        for sf in sfs:
            pre      = f"{args.system}_{benchmark}_"
            suffixes = [t.format(sf) for t in k_suffixes]

            if args.csv:
                files = [(args.csv, "custom")]
            else:
                files = []
                for suf in suffixes:
                    for fpath in glob.glob(os.path.join(args.in_dir, benchmark, f"{pre}*{suf}")):
                        print(f"Checking {fpath} against suffix {suf}...")
                        bname = os.path.basename(fpath)
                        if bname.startswith(pre) and bname.endswith(suf):
                            norm_idx = bname[len(pre):-len(suf)].replace("GPU,", "")
                            files.append((fpath, norm_idx))

            if not files:
                print(f"No files found for benchmark={benchmark}, sf={sf}. Skipping.")
                continue

            # Per-index plots
            for fpath, norm_idx in files:
                print(f"-- Processing {fpath} (Index: {norm_idx}, Bench: {benchmark}, SF: {sf}) --")
                for m in args.metric:
                    all_done.extend(generate_plots(fpath, norm_idx, sf, args.out_dir, m,
                                                   args.system, benchmark))

            # Multi-index plots: one figure per metric with all APPROVED_INDEXES stacked as rows
            for m in args.metric:
                index_entries = []
                for approved_idx in APPROVED_INDEXES:
                    # norm_idx from filenames may use commas (e.g. "IVF1024,Flat") while
                    # APPROVED_INDEXES uses underscores — normalize both sides to compare.
                    match = [(fp, ni) for fp, ni in files
                             if ni.replace(",", "_").replace(" ", "_") == approved_idx]
                    if not match:
                        print(f"  [multi-index] Skipping {approved_idx}: no matching CSV found.")
                        continue
                    fpath, norm_idx = match[0]
                    df = load_data(fpath)
                    metric_ext = f"_{m}"
                    targets = [idx for idx in df.index
                               if str(idx).endswith(metric_ext)
                               and str(idx)[:-len(metric_ext)] not in SKIP_QUERY]
                    if not targets:
                        print(f"  [multi-index] Skipping {approved_idx}: no targets for metric={m}.")
                        continue
                    index_entries.append((df, norm_idx, targets, metric_ext))

                if index_entries:
                    print(f"-- Generating multi-index plot ({len(index_entries)} indexes, metric={m}) --")
                    all_done.extend(generate_multi_index_plots(index_entries, sf, args.out_dir,
                                                               m, args.system, benchmark,
                                                               pg_dir=args.pg_dir, k=args.k))

    if not all_done:
        print("\nError: No plots were generated. Check your --sf and --benchmark arguments.")
        sys.exit(1)

    print(f"\nDONE: {len(all_done)} plots saved.")


if __name__ == "__main__":
    main()
