# Example:
# $ python plot_caliper.py --sf 1 --log --show_values && python plot_caliper.py --sf 1 --show_values
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
    "0: CPU-CPU-CPU": "cpu",
    "1: GPU-CPU-CPU": "mixed", # host + index on host
    "1P: GPU-CPU(P)-CPU(P)": "mixed (pinned)",
    "2: GPU-CPU-GPU": "gpu (h_D)",
    "3: GPU-GPU-GPU": "gpu",
    "4: MIXED(VS=CPU)": "mixed(VS=CPU)",
}

REGIONS = ["Operators", "Data Transfers", "Other"]
RENAME_MAPPING = {
    "Operators": "Operator Execution",
    "Data Transfers": "Data Movement",
    "Other": "Other"
}

SKIP_QUERY = ["q1_start"]
# SKIP_QUERY = []

QUERY_RENAME_MAPPING = {
    "q1_start"    : "Q1",# Start (Reviews - 10xAll)",
    "q2_start"    : "Q2",# Start (Images - 1xMain)",
    "q10_mid"     : "Q10",# Mid (Reviews - 1xAll)",
    "q11_end"     : "Q11",# End (Images - NxMain)",
    "q13_mid"     : "Q13",# Mid (Reviews - 1xAll)",
    "q15_end"     : "Q15",# End (Reviews - 1xAll)",
    "q16_start"   : "Q16",# Start (Reviews - 1xAll)",
    "q18_mid"     : "Q18",# Mid (Reviews -1xMain)",
    "q19_start"   : "Q19",# Start (Images, Reviews - 1xMain, 1xAll)"
}

OPERATORS_TO_PLOT = [
    "VectorSearch", "Filter", "Project", "Join", "GroupBy", 
    "OrderBy", "Limit", #"LimitPerGroup", # optional we can keep it in the limit bucket
    "Take", "Scatter", "Gather",
    "Other"
]

# Style tokens — sized for LaTeX papers (single-col ≈ 3.5", double-col ≈ 7.0")
BAR_WIDTH = 0.6
FIGSIZE_INDIVIDUAL = (3.5, 2.6)       # single-column figure (one query)
FIGSIZE_SUMMARY_WIDTH = 7.0            # double-column figure width
DPI=600

######## 2-4
X_ROTATION=20
USE_ROT=False
LEG_BUFFER_1=0.08
LEG_BUFFER_2=0.08
SUMMARY_GRID_ROWS = 2                 # fixed 2-row × 4-col layout for 8-query VSDS grids
SUMMARY_GRID_COLS = 4
FIGSIZE_SUMMARY_HEIGHT_PER_ROW = 2.25 # per-row height for summary grids (2 rows → 4.5" total)

TITLE_FONTSIZE   = 8   # ax.set_title() — subplot titles: query names ("Q1") and case labels ("cpu") in grids
LABEL_FONTSIZE   = 8   # ax.set_ylabel() / ax.set_xlabel() — axis labels ("Runtime [ms]", "Case")
XTICK_FONTSIZE   = 8   # ax.set_xticklabels() — x-axis tick-mark labels (category names, run numbers)
YTICK_FONTSIZE   = 6  # ax.tick_params(axis='y') — y-axis tick-mark numbers
VALUE_FONTSIZE   = 8   # ax.text() — numeric value annotations printed on top of bars
LEGEND_FONTSIZE  = 8   # ax.legend() / fig.legend() — legend text

# # ######### 1-8
# X_ROTATION=30
# LEG_BUFFER_1=0.15
# LEG_BUFFER_2=0.22
# USE_ROT=True
# SUMMARY_GRID_ROWS = 1                 # fixed 2-row × 4-col layout for 8-query VSDS grids
# SUMMARY_GRID_COLS = 8
# FIGSIZE_SUMMARY_HEIGHT_PER_ROW = 1.25 # per-row height for summary grids (2 rows → 4.5" total)

# TITLE_FONTSIZE   = 7  # ax.set_title() — subplot titles: query names ("Q1") and case labels ("cpu") in grids
# LABEL_FONTSIZE   = 7   # ax.set_ylabel() / ax.set_xlabel() — axis labels ("Runtime [ms]", "Case")
# XTICK_FONTSIZE   = 6   # ax.set_xticklabels() — x-axis tick-mark labels (category names, run numbers)
# YTICK_FONTSIZE   = 5  # ax.tick_params(axis='y') — y-axis tick-mark numbers
# VALUE_FONTSIZE   = 6   # ax.text() — numeric value annotations printed on top of bars
# LEGEND_FONTSIZE  = 7   # ax.legend() / fig.legend() — legend text



DEFAULT_FONT_WEIGHT = 'normal'
NORMAL_FONT_WEIGHT = 'normal'
EDGE_COLOR = 'black'

# Color Mapping for consistency
# "Other" stands out with a distinct shade
COLOR_MAPPING = {
    # Plot 1 Categories
    "Operator Execution": "#34495e",  # Dark Blue/Gray
    "Data Movement": "#2ecc71",     # Green
    "Other": "#9b59b6",              # Purple
    
    # Detailed Plot 1 Categories
    "Rel. Operators": "#3498db", # Blue
    "Vector Search":"#e67e22", # Orange
    "Index Movement": "#e74c3c", # Red/Coastal
    
    # Plot 2 Operators (Standard cycle for most, specific for some)
    "VectorSearch": "#e67e22",      # Keep Orange
    "Filter": "#5dade2",            # Sky Blue
    "Project": "#f1c40f",           # Yellow
    "Join": "#ff69b4",              # Pink
    "GroupBy": "#34495e",           # Dark Navy
    "OrderBy": "#c0392b",           # Dark Red
    "Limit": "#2ecc71",             # Green
    # "LimitPerGroup": "#1abc9c",     # Green ( Optional we can also just keep it in the Limit bucket )
    "Take": "#7f8c8d",              # Dark Gray
    "Scatter": "#bdc3c7",           # Light Gray
    "Gather":  "#a29bfe",           # Lavender
    "Other": "#9b59b6"              # Purple (matches "Other" in coarse views)
}

# Default color cycle for any missing operators
DEFAULT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Plot options
SHOW_VALUES = True
SHOW_STD_BARS = True
PLOT_FIXED_Y_LIMITS = False

# Set by main() based on --k argument; appended to all plot titles when non-empty.
_K_LABEL = ""
# Set by main() based on --format argument.
_EXT = ".jpeg"
_K_VALUE = "100"
SHOW_TITLE    = False    # set True via --title to show figure/subplot titles
_PER_QUERY    = True     # set False via --no-per-query to skip individual query plots
_QUICK_MODE   = False    # set True via --quick; only detailed per-query plots, no summaries
_SKIP_CASES   = set()    # set via --skip-cases; default {"2: GPU-CPU-GPU"} added in main()

# ==============================================================================
# 2. CORE UTILITIES (Data Loading & Rendering)
# ==============================================================================

def load_data(csv_file):
    """Reads the CSV with multi-index columns."""
    df = pd.read_csv(csv_file, header=[0, 1], index_col=0)
    return df

def get_plot_path(out_dir, benchmark, sf, system, index_desc, plot_type, metric, filename, per_query=False  ):
    """
    Generates the hierarchy: /plots/<benchmark>/sf_<sf>/<system>/<index>/<plot_type>/<metric>/<filename>
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

def render_stacked_bar(df, title, ylabel, out_path, fig_size=FIGSIZE_INDIVIDUAL,
                       logy=False, legend_title="", ylim=None, rot=15, std_errors=None, xlabel="",
                       legend_outside=False):
    """Unified rendering engine for all stacked bar plots."""
    fig, ax = plt.subplots(figsize=fig_size)
    plot_df = df.fillna(0.0)
    
    # Build color list based on columns
    plot_colors = [COLOR_MAPPING.get(col, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
                   for i, col in enumerate(plot_df.columns)]
    
    plot_df.plot(kind='bar', stacked=True, ax=ax, edgecolor=EDGE_COLOR, width=BAR_WIDTH, logy=logy, color=plot_colors)
    
    if std_errors is not None and SHOW_STD_BARS:
        totals = plot_df.sum(axis=1)
        x_coords = np.arange(len(plot_df))
        if isinstance(std_errors, pd.Series):
            errs = std_errors.reindex(plot_df.index).fillna(0.0).to_numpy()
        else:
            errs = np.asarray(std_errors)
            if errs.ndim == 1 and errs.shape[0] != len(plot_df):
                errs = None
        if errs is not None:
            ax.errorbar(x_coords, totals, yerr=errs, fmt='none', ecolor='black', capsize=6, capthick=2, elinewidth=2, zorder=10)
    
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
    if SHOW_TITLE:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT, pad=20)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rot, ha='right',
                       fontsize=XTICK_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
    ax.tick_params(axis='y', labelsize=YTICK_FONTSIZE)
    
    err_arr = None
    if std_errors is not None and SHOW_STD_BARS:
        if isinstance(std_errors, pd.Series):
            err_arr = std_errors.reindex(plot_df.index).fillna(0.0).to_numpy()
        else:
            tmp = np.asarray(std_errors)
            if tmp.ndim == 1 and tmp.shape[0] == len(plot_df):
                err_arr = np.nan_to_num(tmp)

    if ylim:
        ax.set_ylim(ylim)
    elif SHOW_VALUES:
        # Add some headroom for labels
        curr_min, curr_max = ax.get_ylim()
        
        max_y = curr_max
        if err_arr is not None:
            totals = plot_df.sum(axis=1).values
            max_y_with_err = np.max(totals + err_arr)
            max_y = max(max_y, max_y_with_err)
            
        if logy:
            ax.set_ylim(bottom=curr_min, top=max_y * 5) # Log scale needs more multiplier headroom
        else:
            ax.set_ylim(bottom=curr_min, top=max_y * 1.15)
            
    if SHOW_VALUES:
        totals = plot_df.sum(axis=1, min_count=1).fillna(0.0)
        suffix = ""
        # Get dynamic padding for visual separation
        curr_min, curr_max = ax.get_ylim()
        pad = (curr_max - curr_min) * 0.01

        for i, val in enumerate(totals):
            if not np.isnan(val) and val > 0:
                y_pos = val
                if err_arr is not None and i < len(err_arr):
                    y_pos += err_arr[i]
                
                ax.text(i, y_pos + pad, f"{val:.0f}{suffix}", ha='center', va='bottom',
                        fontweight=DEFAULT_FONT_WEIGHT, fontsize=VALUE_FONTSIZE)

    annotate_missing_cases(ax, df)
        
    if legend_outside or legend_title or len(plot_df.index) <= 2:
        # Place legend outside for small plots (1-2 bars) to avoid overlap
        ax.legend(title=legend_title or None, loc='center left', bbox_to_anchor=(1.0, 0.5),
                  frameon=False, prop={'weight': DEFAULT_FONT_WEIGHT, 'size': LEGEND_FONTSIZE})
    else:
        ax.legend(loc='upper right', frameon=False, prop={'weight': DEFAULT_FONT_WEIGHT, 'size': LEGEND_FONTSIZE})

    # plt.tight_layout() is suppressed by warnings check
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

# ==============================================================================
# 3. DATA PREPARATION LOGIC
# ==============================================================================

def get_query_data(df, row_idx, detailed=False):
    """Unified data view extractor for Plot 1 variants."""
    row = df.loc[row_idx]
    df_plot = row.unstack(level=0)
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
        # Detailed split logic
        ops, vs = get_col('Operators').fillna(0.0), get_col('VectorSearch').fillna(0.0)
        dt, im  = get_col('Data Transfers').fillna(0.0), get_col('IndexMovement').fillna(0.0)
        oth     = get_col('Other').fillna(0.0)

        df_plot['Rel. Operators'] = (ops - vs).clip(lower=0)
        df_plot['Vector Search'] = vs
        df_plot['Data Movement'] = (dt - im).clip(lower=0)
        df_plot['Index Movement'] = im
        df_plot['Other'] = oth

        detailed_cols = ["Rel. Operators", "Vector Search", "Data Movement", "Index Movement", "Other"]
        plot_data = df_plot[detailed_cols].copy()
        plot_data.loc[missing_case_mask, :] = np.nan
    
    return plot_data

# ==============================================================================
# 4. PLOTTING SUITE: PLOT 1 (Breakdowns & Aggregates)
# ==============================================================================

def plot_1_individual(df, target_indices, metric_suffix, index_desc, sf, out_dir, metric, system, benchmark, detailed=False, fixed_ylim=None):
    """Generates standard or detailed individual query plots."""
    suffix_tag = "detailed_" if detailed else ""
    plot_name = "plot_1"
    generated = []

    for row_idx in target_indices:
        q = str(row_idx)[:-len(metric_suffix)]
        name = QUERY_RENAME_MAPPING.get(q, q)
        plot_data = get_query_data(df, row_idx, detailed=detailed)
        
        std_errors = None
        if metric in ["mean"] and SHOW_STD_BARS:
            std_row_idx = f"{q}_std"
            if std_row_idx in df.index:
                std_df = df.loc[std_row_idx].unstack(level=0).rename(index=CASE_RENAME_MAPPING)
                if 'Total' in std_df.columns:
                    std_errors = std_df['Total'].reindex(plot_data.index).fillna(0)
        
        f_y = "_fixed_y" if fixed_ylim else ""
        filename = f"{suffix_tag}highlevel_breakdown_{q}{f_y}{_EXT}"
        out_path = get_plot_path(out_dir, benchmark, sf, system, index_desc, plot_name, metric, filename, per_query=True)
        
        title = f"{'Detailed ' if detailed else ''}Query Breakdown: {name}\n({system}, {index_desc}, SF={sf}{_K_LABEL})"
        render_stacked_bar(plot_data, title, "Runtime [ms]", out_path, logy=False, std_errors=std_errors, ylim=fixed_ylim)
        generated.append(out_path)
    return generated

def plot_1_summary_grid(df, target_indices, metric_suffix, index_desc, sf, out_dir, metric, system, benchmark, detailed=False, fixed_ylim=None, suffix=""):
    """Generates grid summaries for Plot 1."""
    if not target_indices: return []
    num_q = len(target_indices)
    rows, cols = SUMMARY_GRID_ROWS, SUMMARY_GRID_COLS

    # Removed layout="tight" to avoid conflicts with custom tight_layout later
    fig, axes = plt.subplots(rows, cols, figsize=(FIGSIZE_SUMMARY_WIDTH, FIGSIZE_SUMMARY_HEIGHT_PER_ROW * rows))
    axes = np.array(axes).reshape(-1) if num_q > 1 else [axes]
    
    for i, row_idx in enumerate(target_indices):
        q = str(row_idx)[:-len(metric_suffix)]
        plot_data = get_query_data(df, row_idx, detailed=detailed)
        ax = axes[i]
        
        plot_colors = [COLOR_MAPPING.get(col, DEFAULT_COLORS[j % len(DEFAULT_COLORS)]) 
                       for j, col in enumerate(plot_data.columns)]
        
        plot_data.plot(kind='bar', stacked=True, ax=ax, edgecolor=EDGE_COLOR, width=BAR_WIDTH, logy=False, color=plot_colors)
        ax.set_title(QUERY_RENAME_MAPPING.get(q, q), fontsize=TITLE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
        if i % SUMMARY_GRID_COLS == 0:
            ax.set_ylabel("Runtime [ms]", fontweight=DEFAULT_FONT_WEIGHT, fontsize=LABEL_FONTSIZE)
        # set fontweight for x axis ticks:
        if not USE_ROT:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=X_ROTATION, fontsize=XTICK_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT) # optional: rotation=20, ha='right',
        else:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=X_ROTATION , ha='right', fontsize=XTICK_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT) # optional: rotation=20, ha='right',
        
        ax.tick_params(axis='x', pad=0)
        ax.tick_params(axis='y', labelsize=YTICK_FONTSIZE)
        if ax.get_legend(): ax.get_legend().remove()
        
        # --- NEW CODE: Remove top and right spines to reduce clutter ---
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Optional: Make the remaining axis lines slightly lighter so the data pops
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
                    ax.errorbar(x_coords, totals, yerr=errs, fmt='none', ecolor='black', capsize=6, capthick=2, elinewidth=2, zorder=10)
                    err_arr = errs.values

        if fixed_ylim:
            ax.set_ylim(fixed_ylim)

        elif SHOW_VALUES:
            totals = plot_data.sum(axis=1, min_count=1).fillna(0.0)
            curr_min, curr_max = ax.get_ylim()
            
            max_y = curr_max
            if err_arr is not None:
                max_y_with_err = np.max(totals.values + err_arr)
                max_y = max(max_y, max_y_with_err)
                
            # Reduced headroom to 10% (1.10) so bars take up more relative space
            ax.set_ylim(bottom=curr_min, top=max_y * 1.10)
            
        if SHOW_VALUES:
            totals = plot_data.sum(axis=1, min_count=1).fillna(0.0)
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

    # Position Legend at the bottom center of the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False,
               prop={'weight': NORMAL_FONT_WEIGHT, 'size': LEGEND_FONTSIZE })

    top_margin = 1
    if SHOW_TITLE:
        tag = "Detailed " if detailed else ""
        title = f"{tag}Summary Breakdown ({metric.upper()}) - {system} ({index_desc}, SF={sf}{_K_LABEL})"
        plt.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
        top_margin = 0.92 # Leave space for the title

    # Force a custom layout: [left, bottom, right, top]
    # bottom=0.15 leaves the bottom 2% of the figure completely empty for the legend
    # fig.tight_layout(rect=[0, 0.02, 1, top_margin])
    # fig.tight_layout(rect=[0, 0.02, 1, 0.1], w_pad=0, h_pad=0)
    fig.tight_layout(pad=0, w_pad=0.2, rect=[0, LEG_BUFFER_1, 1, 1.0])
    
    prefix = "detailed_" if detailed else ""
    f_y = "_fixed_y" if fixed_ylim else ""
    filename = f"{prefix}summary_highlevel_breakdown{f_y}{suffix}{_EXT}"
    out_path = get_plot_path(out_dir, benchmark, sf, system, index_desc, "plot_1", metric, filename)
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight'); plt.close(fig)
    return [out_path]

def plot_1_aggregate(df, target_indices, metric_suffix, index_desc, sf, out_dir, metric, system, benchmark, detailed=False, suffix=""):
    """Generates aggregate sum plots for Plot 1."""
    if not target_indices: return []
    all_rows = [get_query_data(df, ridx, detailed) for ridx in target_indices]
    agg_df = pd.concat(all_rows).groupby(level=0).sum()
    
    # Sort cases to match mapping order
    cases_in_order = [CASE_RENAME_MAPPING[k] for k in CASE_RENAME_MAPPING if CASE_RENAME_MAPPING[k] in agg_df.index]
    agg_df = agg_df.reindex(cases_in_order)
    
    std_errors = None
    if metric in ["mean"] and SHOW_STD_BARS:
        # For aggregated values of independent variables, the proper mathematical 
        # approach is to sum their variances (std^2), and take the square root 
        # of the total sum: SD_total = sqrt(sum(SD_i^2))
        var_list = []
        for ridx in target_indices:
            q = str(ridx)[:-len(metric_suffix)]
            std_row_idx = f"{q}_std"
            if std_row_idx in df.index:
                std_df = df.loc[std_row_idx].unstack(level=0).rename(index=CASE_RENAME_MAPPING)
                if 'Total' in std_df.columns:
                    var_list.append(std_df['Total']**2) # store variance
        if var_list:
            sum_of_variances = pd.concat(var_list, axis=1).sum(axis=1).reindex(agg_df.index).fillna(0)
            std_errors = np.sqrt(sum_of_variances) # sqrt(sum of variances)
    
    tag = "Detailed " if detailed else ""
    prefix = "detailed_" if detailed else ""
    filename = f"aggregate_{prefix}highlevel_breakdown{suffix}{_EXT}"
    out_path = get_plot_path(out_dir, benchmark, sf, system, index_desc, "plot_1", metric, filename)
    title = f"Aggregate {tag}Breakdown (Sum of Queries)\n({system}, {index_desc}, SF={sf}{_K_LABEL}, {metric.upper()})"
    
    render_stacked_bar(agg_df, title, "Total Runtime [ms]", out_path, logy=False, std_errors=std_errors)
    return [out_path]

# ==============================================================================
# 5. PLOTTING SUITE: PLOT 2 (Operator Level)
# ==============================================================================

def get_operator_data(df, target_indices, case_label, metric_suffix, coarse=False):
    """Helper to extract operator or vector-search-vs-relational data rows."""
    rows, q_names = [], []
    for ridx in target_indices:
        q = str(ridx)[:-len(metric_suffix)]
        q_names.append(QUERY_RENAME_MAPPING.get(q, q))
        row_data = df.loc[ridx]
        
        if not coarse:
            data = {op: row_data.get((op, case_label), np.nan) for op in OPERATORS_TO_PLOT}
            has_any = any(pd.notna(v) for v in data.values())
            if has_any:
                data = {k: (0.0 if pd.isna(v) else v) for k, v in data.items()}
        else:
            vs_raw = row_data.get(("VectorSearch", case_label), np.nan)
            dc_raw = row_data.get(("Other", case_label), np.nan)
            rel_ops_raw = [row_data.get((op, case_label), np.nan)
                           for op in OPERATORS_TO_PLOT if op not in ["VectorSearch", "Other"]]
            if pd.isna(vs_raw) and pd.isna(dc_raw) and all(pd.isna(v) for v in rel_ops_raw):
                data = {"Vector Search": np.nan, "Rel. Operators": np.nan, "Other": np.nan}
            else:
                vs_val = 0.0 if pd.isna(vs_raw) else vs_raw
                dc_val = 0.0 if pd.isna(dc_raw) else dc_raw
                rel_val = float(np.nansum(rel_ops_raw))
                data = {"Vector Search": vs_val, "Rel. Operators": rel_val, "Other": dc_val}
            
        rows.append(data)
    return pd.DataFrame(rows, index=q_names)

def get_query_operator_data(df, row_idx, coarse=False):
    """Extract operator breakdown for a single query across all execution cases."""
    row_data = df.loc[row_idx]
    cases = [c for c in CASE_RENAME_MAPPING.keys() if c not in _SKIP_CASES]
    plot_rows = []

    for case_label in cases:
        if not coarse:
            data = {op: row_data.get((op, case_label), np.nan) for op in OPERATORS_TO_PLOT}
            has_any = any(pd.notna(v) for v in data.values())
            if has_any:
                data = {k: (0.0 if pd.isna(v) else v) for k, v in data.items()}
        else:
            vs_raw = row_data.get(("VectorSearch", case_label), np.nan)
            dc_raw = row_data.get(("Other", case_label), np.nan)
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

def plot_2_individual(df, target_indices, metric_suffix, index_desc, sf, out_dir, metric, system, benchmark):
    """Generates operator breakdown plots for each individual query."""
    generated = []
    plot_name = "plot_2"

    for row_idx in target_indices:
        q = str(row_idx)[:-len(metric_suffix)]
        name = QUERY_RENAME_MAPPING.get(q, q)
        
        # Standard Plot
        plot_df = get_query_operator_data(df, row_idx, coarse=False)
        
        std_errors = None
        if metric in ["mean"] and SHOW_STD_BARS:
            std_row_idx = f"{q}_std"
            if std_row_idx in df.index:
                std_df = df.loc[std_row_idx].unstack(level=0)
                if 'Operators' in std_df.columns:
                    # Filter and reindex correctly based on cases in plot_df
                    cases = list(CASE_RENAME_MAPPING.keys())
                    std_errors = std_df['Operators'].reindex(cases).fillna(0)
                    std_errors.index = [CASE_RENAME_MAPPING[c] for c in cases]
                    std_errors = std_errors.reindex(plot_df.index).fillna(0)
        
        filename = f"operator_breakdown_{q}{_EXT}"
        out_path = get_plot_path(out_dir, benchmark, sf, system, index_desc, plot_name, metric, filename, per_query=True)
        title = f"Operator Breakdown: {name}\n({system}, {index_desc}, SF={sf}{_K_LABEL}, {metric.upper()})"
        render_stacked_bar(plot_df, title, "Runtime [ms]", out_path,
                           logy=False, legend_title="Operators", std_errors=std_errors)
        generated.append(out_path)

        # Coarse Plot
        plot_df_coarse = get_query_operator_data(df, row_idx, coarse=True)
        std_errors_coarse = std_errors.reindex(plot_df_coarse.index).fillna(0) if isinstance(std_errors, pd.Series) else std_errors
        filename_coarse = f"operator_coarse_breakdown_{q}{_EXT}"
        out_path_coarse = get_plot_path(out_dir, benchmark, sf, system, index_desc, plot_name, metric, filename_coarse, per_query=True)
        title_coarse = f"Coarse Operator Breakdown: {name}\n({system}, {index_desc}, SF={sf}{_K_LABEL}, {metric.upper()})"
        render_stacked_bar(plot_df_coarse, title_coarse, "Runtime [ms]", out_path_coarse,
                   logy=False, legend_title="Operators", std_errors=std_errors_coarse)
        generated.append(out_path_coarse)
        
    return generated

def plot_2_case_comparison(df, target_indices, metric_suffix, index_desc, sf, out_dir, metric, system, benchmark):
    """Universal generator for Plot 2 variants (Standard, Coarse)."""
    generated = []
    configs = [
        {"coarse": False, "prefix": "operator_breakdown_by_query_", "ylabel": "Runtime [ms]"},
        {"coarse": True,  "prefix": "operator_coarse_breakdown_by_query_", "ylabel": "Runtime [ms]"},
    ]

    for case_label, case_display in CASE_RENAME_MAPPING.items():
        if "1:" in case_label or "2:" in case_label: continue
        
        for cfg in configs:
            plot_df = get_operator_data(df, target_indices, case_label, metric_suffix, cfg["coarse"])
            if plot_df.empty: continue
            
            std_errors = None
            if metric in ["mean"] and SHOW_STD_BARS:
                err_list = []
                for ridx in target_indices:
                    q = str(ridx)[:-len(metric_suffix)]
                    std_row_idx = f"{q}_std"
                    val = 0.0
                    if std_row_idx in df.index:
                        std_df = df.loc[std_row_idx].unstack(level=0)
                        if 'Operators' in std_df.columns and case_label in std_df['Operators'].index:
                            val = std_df['Operators'].loc[case_label]
                    err_list.append(val)
                std_errors = pd.Series(err_list, index=plot_df.index)
            
            case_tag = case_label.replace(' ', '_').replace(':', '')
            filename = f"{cfg['prefix']}{case_tag}{_EXT}"
            out_path = get_plot_path(out_dir, benchmark, sf, system, index_desc, "plot_2", metric, filename)
            
            title = f"Operator {'Coarse ' if cfg['coarse'] else ''}Breakdown: {case_display}\n({system}, {index_desc}, SF={sf}{_K_LABEL}, {metric.upper()})"
            render_stacked_bar(plot_df, title, cfg["ylabel"], out_path, 
                               fig_size=(FIGSIZE_SUMMARY_WIDTH, FIGSIZE_SUMMARY_HEIGHT_PER_ROW), 
                               logy=False, 
                               legend_title="Operators", std_errors=std_errors)
            generated.append(out_path)
    return generated

def plot_2_summary_consolidated(df, target_indices, metric_suffix, index_desc, sf, out_dir, metric, system, benchmark):
    """Generates the side-by-side case comparison summaries for Plot 2."""
    generated = []
    cases = ["0: CPU-CPU-CPU", "3: GPU-GPU-GPU"]
    variants = [
        {"coarse": False, "tag": "operator_breakdown_consolidated", "title": "Operator Breakdown Summary"},
        {"coarse": True,  "tag": "operator_coarse_breakdown_consolidated", "title": "Coarse Operator Breakdown Summary"},
    ]

    for var in variants:
        plot_rows, x_labels, std_errors_list = [], [], []
        for ridx in target_indices:
            q = str(ridx)[:-len(metric_suffix)]
            q_name = QUERY_RENAME_MAPPING.get(q, q)
            std_row_idx = f"{q}_std"
            std_df = None
            if metric in ["mean"] and SHOW_STD_BARS and std_row_idx in df.index:
                std_df = df.loc[std_row_idx].unstack(level=0)
                
            for clab in cases:
                data = get_operator_data(df, [ridx], clab, metric_suffix, var["coarse"]).iloc[0].to_dict()
                plot_rows.append(data)
                x_labels.append(f"{q_name}\n({'CPU' if 'CPU' in clab else 'GPU'})")
                
                val = 0.0
                if std_df is not None and 'Operators' in std_df.columns and clab in std_df['Operators'].index:
                    val = std_df['Operators'].loc[clab]
                std_errors_list.append(val)
                
            # Spacer
            plot_rows.append({k: np.nan for k in plot_rows[-1]})
            x_labels.append("")
            std_errors_list.append(0.0)

        pdf = pd.DataFrame(plot_rows, index=x_labels)
        std_errors = pd.Series(std_errors_list, index=pdf.index) if (metric in ["mean"] and SHOW_STD_BARS) else None
        
        filename = f"{var['tag']}{_EXT}"
        out_path = get_plot_path(out_dir, benchmark, sf, system, index_desc, "plot_2", metric, filename)
        ylabel = "Runtime [ms]"
        
        dyn_w = max(FIGSIZE_SUMMARY_WIDTH, len(plot_rows) * 0.8)
        title = f"{var['title']} - {system} ({index_desc}, SF={sf}{_K_LABEL})"
        render_stacked_bar(pdf, title, ylabel, out_path, 
                            fig_size=(dyn_w, FIGSIZE_SUMMARY_HEIGHT_PER_ROW),
                            logy=False, rot=30,
                            legend_title="Operators", std_errors=std_errors)
        generated.append(out_path)
    return generated

def plot_2_summary_grid(df, target_indices, metric_suffix, index_desc, sf, out_dir, metric, system, benchmark, coarse=False, suffix=""):
    """Generates grid summaries for Plot 2 (one subplot per query, operator breakdown across cases)."""
    if not target_indices: return []
    num_q = len(target_indices)
    rows, cols = SUMMARY_GRID_ROWS, SUMMARY_GRID_COLS

    # Removed layout="tight"
    fig, axes = plt.subplots(rows, cols, figsize=(FIGSIZE_SUMMARY_WIDTH, FIGSIZE_SUMMARY_HEIGHT_PER_ROW * rows))
    axes = np.array(axes).reshape(-1) if num_q > 1 else [axes]

    for i, row_idx in enumerate(target_indices):
        q = str(row_idx)[:-len(metric_suffix)]
        plot_data = get_query_operator_data(df, row_idx, coarse=coarse)
        ax = axes[i]

        plot_colors = [COLOR_MAPPING.get(col, DEFAULT_COLORS[j % len(DEFAULT_COLORS)])
                       for j, col in enumerate(plot_data.columns)]

        plot_data.plot(kind='bar', stacked=True, ax=ax, edgecolor=EDGE_COLOR, width=BAR_WIDTH, logy=False, color=plot_colors)
        ax.set_title(QUERY_RENAME_MAPPING.get(q, q), fontsize=TITLE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
        if i % SUMMARY_GRID_COLS == 0:
            ax.set_ylabel("Runtime [ms]", fontweight=DEFAULT_FONT_WEIGHT, fontsize=LABEL_FONTSIZE)
        if not USE_ROT:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=X_ROTATION, fontsize=XTICK_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=X_ROTATION, ha='right', fontsize=XTICK_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
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
                    ax.errorbar(x_coords, totals, yerr=errs, fmt='none', ecolor='black', capsize=6, capthick=2, elinewidth=2, zorder=10)
                    err_arr = errs.values

        if SHOW_VALUES:
            totals = plot_data.sum(axis=1, min_count=1).fillna(0.0)
            curr_min, curr_max = ax.get_ylim()
            max_y = curr_max
            if err_arr is not None:
                max_y_with_err = np.max(totals.values + err_arr)
                max_y = max(max_y, max_y_with_err)
            
            # Reduced headroom to 1.10
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
    fig.legend(handles, labels, loc='lower center', ncol=math.ceil(len(labels) / 2), frameon=False,
               prop={'weight': NORMAL_FONT_WEIGHT, 'size': LEGEND_FONTSIZE})

    top_margin = 1
    if SHOW_TITLE:
        tag = "Coarse " if coarse else ""
        title = f"{tag}Operator Breakdown Summary ({metric.upper()}) - {system} ({index_desc}, SF={sf}{_K_LABEL})"
        plt.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
        top_margin = 0.92

    fig.tight_layout(pad=0, w_pad=0.2, rect=[0, LEG_BUFFER_2, 1, 1.0])

    prefix = "coarse_" if coarse else ""
    filename = f"operator_{'coarse_' if coarse else ''}breakdown_grid{suffix}{_EXT}"
    out_path = get_plot_path(out_dir, benchmark, sf, system, index_desc, "plot_2", metric, filename)
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight'); plt.close(fig)
    return [out_path]


def plot_2_aggregate(df, target_indices, metric_suffix, index_desc, sf, out_dir, metric, system, benchmark):
    """Generates aggregate sum plots for Plot 2 (Operator Level)."""
    generated = []
    configs = [
        {"coarse": False, "tag": "operator_breakdown_aggregate", "ylabel": "Runtime [ms]"},
        {"coarse": True,  "tag": "operator_coarse_breakdown_aggregate", "ylabel": "Runtime [ms]"},
    ]

    for cfg in configs:
        agg_rows = []
        labels = []
        err_rows = []
        for case_label, case_display in CASE_RENAME_MAPPING.items():
            if "1:" in case_label or "2:" in case_label: continue
            
            plot_df = get_operator_data(df, target_indices, case_label, metric_suffix, cfg["coarse"])
            if plot_df.empty: continue
            
            summed = plot_df.sum()
            agg_rows.append(summed)
            labels.append(case_display)
            
            total_variance = 0.0
            if metric in ["mean"] and SHOW_STD_BARS:
                # Calculate SD_total = sqrt(sum(SD_i^2)) for independent queries
                for ridx in target_indices:
                    q = str(ridx)[:-len(metric_suffix)]
                    std_row_idx = f"{q}_std"
                    if std_row_idx in df.index:
                        std_df = df.loc[std_row_idx].unstack(level=0)
                        if 'Operators' in std_df.columns and case_label in std_df['Operators'].index:
                            total_variance += (std_df['Operators'].loc[case_label])**2
            err_rows.append(np.sqrt(total_variance) if total_variance > 0 else 0.0)
            
        if not agg_rows: continue
        
        pdf = pd.DataFrame(agg_rows, index=labels)
        std_errors = pd.Series(err_rows, index=labels) if (metric in ["mean"] and SHOW_STD_BARS) else None
        
        filename = f"{cfg['tag']}{_EXT}"
        out_path = get_plot_path(out_dir, benchmark, sf, system, index_desc, "plot_2", metric, filename)
        
        title = f"Aggregate Operator {'Coarse ' if cfg['coarse'] else ''}Breakdown\n({system}, {index_desc}, SF={sf}{_K_LABEL}, {metric.upper()})"
            
        render_stacked_bar(pdf, title, cfg["ylabel"], out_path, 
                           logy=False, 
                           legend_title="Operators", std_errors=std_errors)
        generated.append(out_path)
    return generated

# ==============================================================================
# 6. PER-REP PLOTS (Plot 3 & Plot 4)
# ==============================================================================

def get_per_rep_query_data(df, query_name, case_label, detailed=True):
    """Extract per-rep detailed breakdown for a single query+case. Returns DataFrame with Run #N index."""
    import re as _re
    pattern = _re.compile(rf'^{_re.escape(query_name)}_rep(\d+)$')
    matching = [(idx, int(pattern.match(str(idx)).group(1))) for idx in df.index if pattern.match(str(idx))]
    if not matching:
        return pd.DataFrame()

    matching.sort(key=lambda x: x[1])
    rows = []
    run_labels = []
    for idx, rep_num in matching:
        row_data = df.loc[idx]

        ops = row_data.get(("Operators", case_label), 0.0) or 0.0
        vs = row_data.get(("VectorSearch", case_label), 0.0) or 0.0
        dt = row_data.get(("Data Transfers", case_label), 0.0) or 0.0
        im = row_data.get(("IndexMovement", case_label), 0.0) or 0.0
        oth = row_data.get(("Other", case_label), 0.0) or 0.0

        if detailed:
            data = {
                "Rel. Operators": max(ops - vs, 0.0),
                "Vector Search": vs,
                "Data Movement": max(dt - im, 0.0),
                "Index Movement": im,
                "Other": oth,
            }
        else:
            data = {
                "Operator Execution": ops,
                "Data Movement": dt,
                "Other": oth,
            }
        rows.append(data)
        run_labels.append(f"{rep_num}")

    return pd.DataFrame(rows, index=run_labels)


def get_per_rep_operator_data(df, query_name, case_label, coarse=False):
    """Extract per-rep operator breakdown for a single query+case."""
    import re as _re
    pattern = _re.compile(rf'^{_re.escape(query_name)}_rep(\d+)$')
    matching = [(idx, int(pattern.match(str(idx)).group(1))) for idx in df.index if pattern.match(str(idx))]
    if not matching:
        return pd.DataFrame()

    matching.sort(key=lambda x: x[1])
    rows = []
    run_labels = []
    for idx, rep_num in matching:
        row_data = df.loc[idx]
        if not coarse:
            data = {op: (row_data.get((op, case_label), 0.0) or 0.0) for op in OPERATORS_TO_PLOT}
        else:
            vs_val = row_data.get(("VectorSearch", case_label), 0.0) or 0.0
            dc_val = row_data.get(("Other", case_label), 0.0) or 0.0
            rel_val = sum((row_data.get((op, case_label), 0.0) or 0.0) for op in OPERATORS_TO_PLOT if op not in ["VectorSearch", "Other"])
            data = {"Vector Search": vs_val, "Rel. Operators": rel_val, "Other": dc_val}
        rows.append(data)
        run_labels.append(f"{rep_num}")

    return pd.DataFrame(rows, index=run_labels)


def plot_3_individual(df_per_rep, queries, case_label, case_tag, index_desc, sf, out_dir, system, benchmark):
    """Plot 3: Per-rep detailed breakdown. One plot per (query, case)."""
    generated = []
    for q in queries:
        if q in SKIP_QUERY:
            continue
        plot_data = get_per_rep_query_data(df_per_rep, q, case_label, detailed=True)
        if plot_data.empty:
            continue

        case_display = CASE_RENAME_MAPPING.get(case_label, case_label)
        name = QUERY_RENAME_MAPPING.get(q, q)
        title = f"Per-Run Detailed Breakdown: {name}\n({case_display}, {system}, {index_desc}, SF={sf}{_K_LABEL})"

        c_tag = case_tag.replace(' ', '_').replace(':', '')
        safe_idx = index_desc.replace(",", "_").replace(" ", "_")
        filename = f"{system}_{benchmark}_sf{sf}_k{_K_VALUE}_{safe_idx}_perrep_highlevel_breakdown_{q}_{c_tag}{_EXT}"
        plot_dir = os.path.join(out_dir, benchmark, f"sf_{sf}", system, index_desc, "plot_3", "per_rep", "per_query")
        os.makedirs(plot_dir, exist_ok=True)
        out_path = os.path.join(plot_dir, filename)

        render_stacked_bar(plot_data, title, "Runtime [ms]", out_path, rot=0, xlabel="Run",
                           legend_outside=True)
        generated.append(out_path)
    return generated


def plot_3_summary(df_per_rep, query, index_desc, sf, out_dir, system, benchmark):
    """Plot 3 summary: 1x4 grid (one subplot per case) for a single query."""
    if query in SKIP_QUERY:
        return []

    cases = [c for c in CASE_RENAME_MAPPING.keys() if c not in _SKIP_CASES]
    non_empty = [c for c in cases if not get_per_rep_query_data(df_per_rep, query, c, detailed=True).empty]
    if not non_empty:
        return []

    # Compute global y-max across all cases and all runs for this query
    global_max = max(
        get_per_rep_query_data(df_per_rep, query, c, detailed=True).sum(axis=1).max()
        for c in non_empty
    )
    fixed_ylim = (0, global_max * 1.15) if PLOT_FIXED_Y_LIMITS else None

    n_cases = len(non_empty)
    fig, axes = plt.subplots(1, n_cases, figsize=(5 * n_cases, 7))
    if n_cases == 1:
        axes = [axes]

    for i, case_label in enumerate(non_empty):
        plot_data = get_per_rep_query_data(df_per_rep, query, case_label, detailed=True)
        ax = axes[i]
        plot_colors = [COLOR_MAPPING.get(col, DEFAULT_COLORS[j % len(DEFAULT_COLORS)])
                       for j, col in enumerate(plot_data.columns)]
        plot_data.plot(kind='bar', stacked=True, ax=ax, edgecolor=EDGE_COLOR, width=BAR_WIDTH, color=plot_colors)
        ax.set_title(CASE_RENAME_MAPPING.get(case_label, case_label), fontsize=TITLE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
        if i == 0:
            ax.set_ylabel("Runtime [ms]", fontweight=DEFAULT_FONT_WEIGHT, fontsize=LABEL_FONTSIZE)
        ax.set_xlabel("Run", fontsize=LABEL_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=XTICK_FONTSIZE)
        ax.tick_params(axis='y', labelsize=YTICK_FONTSIZE)
        if ax.get_legend():
            ax.get_legend().remove()

        if fixed_ylim:
            ax.set_ylim(fixed_ylim)
        elif SHOW_VALUES:
            curr_min, curr_max = ax.get_ylim()
            ax.set_ylim(bottom=curr_min, top=curr_max * 1.15)

        if SHOW_VALUES:
            totals = plot_data.sum(axis=1)
            pad = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
            for j, val in enumerate(totals):
                if not np.isnan(val) and val > 0:
                    ax.text(j, val + pad, f"{val:.0f}", ha='center', va='bottom',
                            fontweight=DEFAULT_FONT_WEIGHT, fontsize=VALUE_FONTSIZE)

    name = QUERY_RENAME_MAPPING.get(query, query)
    if SHOW_TITLE:
        fig.suptitle(f"Per-Run Detailed Breakdown: {name}\n({system}, {index_desc}, SF={sf}{_K_LABEL})",
                     fontsize=TITLE_FONTSIZE + 4, fontweight=DEFAULT_FONT_WEIGHT)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               frameon=False, prop={'weight': DEFAULT_FONT_WEIGHT, 'size': LEGEND_FONTSIZE})

    # rect=[left, bottom, right, top] reserves space for suptitle (top) and legend (bottom)
    fig.tight_layout(rect=[0, 0.10, 1, 0.93])

    safe_idx = index_desc.replace(",", "_").replace(" ", "_")
    plot_dir = os.path.join(out_dir, benchmark, f"sf_{sf}", system, index_desc, "plot_3", "per_rep")
    os.makedirs(plot_dir, exist_ok=True)
    _fy = "_fixed_y" if PLOT_FIXED_Y_LIMITS else ""
    out_path = os.path.join(plot_dir, f"{system}_{benchmark}_sf{sf}_k{_K_VALUE}_{safe_idx}_perrep_highlevel_breakdown_{query}_all_cases{_fy}{_EXT}")
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return [out_path]


def plot_4_individual(df_per_rep, queries, case_label, case_tag, index_desc, sf, out_dir, system, benchmark, coarse=False):
    """Plot 4: Per-rep operator breakdown. One plot per (query, case)."""
    generated = []
    for q in queries:
        if q in SKIP_QUERY:
            continue
        plot_data = get_per_rep_operator_data(df_per_rep, q, case_label, coarse=coarse)
        if plot_data.empty:
            continue

        case_display = CASE_RENAME_MAPPING.get(case_label, case_label)
        name = QUERY_RENAME_MAPPING.get(q, q)
        coarse_tag = "Coarse " if coarse else ""
        title = f"Per-Run {coarse_tag}Operator Breakdown: {name}\n({case_display}, {system}, {index_desc}, SF={sf}{_K_LABEL})"

        c_tag = case_tag.replace(' ', '_').replace(':', '')
        safe_idx = index_desc.replace(",", "_").replace(" ", "_")
        filename = f"{system}_{benchmark}_sf{sf}_k{_K_VALUE}_{safe_idx}_perrep_operator_{'coarse_' if coarse else ''}breakdown_{q}_{c_tag}{_EXT}"
        plot_dir = os.path.join(out_dir, benchmark, f"sf_{sf}", system, index_desc, "plot_4", "per_rep", "per_query")
        os.makedirs(plot_dir, exist_ok=True)
        out_path = os.path.join(plot_dir, filename)

        render_stacked_bar(plot_data, title, "Runtime [ms]", out_path, rot=0, legend_title="Operators",
                           xlabel="Run", legend_outside=True)
        generated.append(out_path)
    return generated


def plot_4_summary(df_per_rep, query, index_desc, sf, out_dir, system, benchmark, coarse=False):
    """Plot 4 summary: 1x4 grid (one subplot per case) for a single query, operator breakdown."""
    if query in SKIP_QUERY:
        return []

    cases = [c for c in CASE_RENAME_MAPPING.keys() if c not in _SKIP_CASES]
    non_empty = [c for c in cases if not get_per_rep_operator_data(df_per_rep, query, c, coarse=coarse).empty]
    if not non_empty:
        return []

    # Compute global y-max across all cases and all runs for this query
    global_max = max(
        get_per_rep_operator_data(df_per_rep, query, c, coarse=coarse).sum(axis=1).max()
        for c in non_empty
    )
    fixed_ylim = (0, global_max * 1.15) if PLOT_FIXED_Y_LIMITS else None

    n_cases = len(non_empty)
    fig, axes = plt.subplots(1, n_cases, figsize=(5 * n_cases, 7))
    if n_cases == 1:
        axes = [axes]

    for i, case_label in enumerate(non_empty):
        plot_data = get_per_rep_operator_data(df_per_rep, query, case_label, coarse=coarse)
        ax = axes[i]
        plot_colors = [COLOR_MAPPING.get(col, DEFAULT_COLORS[j % len(DEFAULT_COLORS)])
                       for j, col in enumerate(plot_data.columns)]
        plot_data.plot(kind='bar', stacked=True, ax=ax, edgecolor=EDGE_COLOR, width=BAR_WIDTH, color=plot_colors)
        ax.set_title(CASE_RENAME_MAPPING.get(case_label, case_label), fontsize=TITLE_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
        if i == 0:
            ax.set_ylabel("Runtime [ms]", fontweight=DEFAULT_FONT_WEIGHT, fontsize=LABEL_FONTSIZE)
        ax.set_xlabel("Run", fontsize=LABEL_FONTSIZE, fontweight=DEFAULT_FONT_WEIGHT)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=XTICK_FONTSIZE)
        ax.tick_params(axis='y', labelsize=YTICK_FONTSIZE)
        if ax.get_legend():
            ax.get_legend().remove()

        if fixed_ylim:
            ax.set_ylim(fixed_ylim)
        elif SHOW_VALUES:
            curr_min, curr_max = ax.get_ylim()
            ax.set_ylim(bottom=curr_min, top=curr_max * 1.15)

        if SHOW_VALUES:
            totals = plot_data.sum(axis=1)
            pad = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01
            for j, val in enumerate(totals):
                if not np.isnan(val) and val > 0:
                    ax.text(j, val + pad, f"{val:.0f}", ha='center', va='bottom',
                            fontweight=DEFAULT_FONT_WEIGHT, fontsize=VALUE_FONTSIZE)

    coarse_tag = "Coarse " if coarse else ""
    name = QUERY_RENAME_MAPPING.get(query, query)
    if SHOW_TITLE:
        fig.suptitle(f"Per-Run {coarse_tag}Operator Breakdown: {name}\n({system}, {index_desc}, SF={sf}{_K_LABEL})",
                     fontsize=TITLE_FONTSIZE + 4, fontweight=DEFAULT_FONT_WEIGHT)

    handles, labels = axes[0].get_legend_handles_labels()
    # ncol=ceil(n/2) means 2 legend rows for many operators — reserve more bottom space than plot 3
    fig.legend(handles, labels, loc='lower center', ncol=math.ceil(len(labels) / 2),
               frameon=False, prop={'weight': DEFAULT_FONT_WEIGHT, 'size': LEGEND_FONTSIZE})

    # rect=[left, bottom, right, top]: extra bottom for 2-row legend, top for suptitle
    fig.tight_layout(rect=[0, 0.17, 1, 0.93])

    safe_idx = index_desc.replace(",", "_").replace(" ", "_")
    plot_dir = os.path.join(out_dir, benchmark, f"sf_{sf}", system, index_desc, "plot_4", "per_rep")
    os.makedirs(plot_dir, exist_ok=True)
    _fy = "_fixed_y" if PLOT_FIXED_Y_LIMITS else ""
    out_path = os.path.join(plot_dir, f"{system}_{benchmark}_sf{sf}_k{_K_VALUE}_{safe_idx}_perrep_operator_{'coarse_' if coarse else ''}breakdown_{query}_all_cases{_fy}{_EXT}")
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return [out_path]


def generate_per_rep_plots(per_rep_csv, index_desc, sf, out_dir, system, benchmark, plots=None):
    """Generate Plot 3 and Plot 4 from per-rep CSV data."""
    df_per_rep = load_data(per_rep_csv)
    all_p = []
    want = plots  # None means all

    # Extract unique query names from index (strip _repN suffix)
    import re as _re
    queries = sorted(set(_re.sub(r'_rep\d+$', '', str(idx)) for idx in df_per_rep.index))

    for case_label in [c for c in CASE_RENAME_MAPPING.keys() if c not in _SKIP_CASES]:
        case_tag = case_label.replace(' ', '_').replace(':', '')
        if want is not None and 3 in want:
            all_p += plot_3_individual(df_per_rep, queries, case_label, case_tag, index_desc, sf, out_dir, system, benchmark)
        if want is not None and 4 in want:
            all_p += plot_4_individual(df_per_rep, queries, case_label, case_tag, index_desc, sf, out_dir, system, benchmark, coarse=False)

    for q in queries:
        if want is not None and 3 in want:
            all_p += plot_3_summary(df_per_rep, q, index_desc, sf, out_dir, system, benchmark)
        if want is not None and 4 in want:
            all_p += plot_4_summary(df_per_rep, q, index_desc, sf, out_dir, system, benchmark, coarse=False)

    return all_p


# ==============================================================================
# 7. ORCHESTRATION & MAIN
# ==============================================================================

def generate_plots(csv_file, index_desc, sf, out_dir, metric, system, benchmark, per_rep_csv=None, plots=None):
    """Unified entry point for generating the full suite of VSDS plots."""
    global _SKIP_CASES
    df = load_data(csv_file)
    metric_ext = f"_{metric}"
    
    # Global Filter for SKIP_QUERY
    targets = [idx for idx in df.index if str(idx).endswith(metric_ext) and str(idx)[:-len(metric_ext)] not in SKIP_QUERY]
    metric_df = df[df.index.isin(targets)]
    
    # Calculate global max for Plot 1 fixed-y scaling
    # Sum only the primary data regions to get the true total per case
    # metric_df columns are (Case, Region)
    available_regions = [r for r in REGIONS if r in metric_df.columns.get_level_values(0)]

    if not available_regions or metric_df.empty:
        global_max_val = 100 # Default fallback
    else:
        # Calculate totals per query per case
        # Summing across the Region level (level 1)
        # Fix: avoid deprecated groupby(axis=1)
        case_totals = metric_df.loc[:, (available_regions, slice(None))].T.groupby(level=1).sum().T
        global_max_val = case_totals.max().max()
    
    # Also consider STD if shown (approximate by adding the max total std)
    if metric == "mean" and SHOW_STD_BARS:
        std_targets = [f"{str(idx)[:-len(metric_ext)]}_std" for idx in targets]
        std_df = df[df.index.isin(std_targets)]
        if not std_df.empty:
            # We want the max of the 'Total' column in std_df if it exists
            if 'Total' in std_df.columns.get_level_values(0):
                max_std = std_df.xs('Total', axis=1, level=0).max().max()
                if not np.isnan(max_std):
                    global_max_val += max_std
    
    # Final safety check for NaN
    if np.isnan(global_max_val) or global_max_val <= 0:
        global_max_val = 100
        
    fixed_ylim = (0, global_max_val * 1.15)
    
    want = plots  # None means all

    all_p = []

    def _annotate_segments(ax, plot_data, x_idx=0, fontsize=9):
        """Annotate each stacked segment with 'Name: X ms (Y%)' to the right of the bar."""
        total = plot_data.iloc[x_idx].sum()
        if total <= 0:
            return
        bottom = 0
        bar_right = x_idx + BAR_WIDTH / 2
        annotations = []
        for col in plot_data.columns:
            val = plot_data.iloc[x_idx][col]
            if val <= 0:
                bottom += val
                continue
            pct = val / total * 100
            if val > 0:  # annotate all non-zero segments
                center_y = bottom + val / 2
                label = f"{col}: {val:.1f} ms ({pct:.0f}%)"
                annotations.append((center_y, label))
            bottom += val
        # Space annotations to avoid overlap (min gap = 3% of y range)
        y_min, y_max = ax.get_ylim()
        min_gap = (y_max - y_min) * 0.03
        # Sort by y position and adjust if overlapping
        annotations.sort(key=lambda a: a[0])
        adjusted = []
        for cy, lbl in annotations:
            if adjusted and cy - adjusted[-1][0] < min_gap:
                cy = adjusted[-1][0] + min_gap
            adjusted.append((cy, lbl))
        for cy, lbl in adjusted:
            ax.annotate(lbl, xy=(bar_right, cy),
                        xytext=(bar_right + 0.35, cy),
                        fontsize=fontsize, va='center',
                        arrowprops=dict(arrowstyle='-', color='#333333', lw=1.0, shrinkA=0, shrinkB=2),
                        fontweight='normal')

    # Quick mode: only detailed per-query plots for the case(s) with data.
    # No summaries, grids, aggregates, or coarse versions.
    if _QUICK_MODE:
        for row_idx in targets:
            q = str(row_idx)[:-len(metric_ext)]
            name = QUERY_RENAME_MAPPING.get(q, q)

            # Font sizes for quick mode (larger than grid defaults)
            _QF_TITLE = 11
            _QF_LABEL = 11
            _QF_XTICK = 10
            _QF_YTICK = 9
            _QF_VALUE = 10
            _QF_LEGEND = 9
            _QF_ANNOT = 9

            def _quick_plot(plot_data, title, out_path, legend_title="", figsize=None):
                """Render a single-case stacked bar with segment annotations."""
                n_cols = len([c for c in plot_data.columns if plot_data[c].sum() > 0])
                default_w = 5.5 if n_cols <= 6 else 7
                fig, ax = plt.subplots(figsize=figsize or (default_w, 4.5))
                colors = [COLOR_MAPPING.get(c, DEFAULT_COLORS[ci % len(DEFAULT_COLORS)]) for ci, c in enumerate(plot_data.columns)]
                plot_data.plot(kind='bar', stacked=True, ax=ax, edgecolor=EDGE_COLOR, width=BAR_WIDTH, color=colors)
                ax.set_ylabel("Runtime [ms]", fontsize=_QF_LABEL, fontweight=DEFAULT_FONT_WEIGHT)
                ax.set_title(title, fontsize=_QF_TITLE, fontweight=DEFAULT_FONT_WEIGHT, pad=20)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right', fontsize=_QF_XTICK)
                ax.tick_params(axis='y', labelsize=_QF_YTICK)
                ax.set_axisbelow(True)
                ax.grid(axis='both', linestyle='--', alpha=0.4)
                ax.legend(title=legend_title or None, loc='upper left', frameon=False,
                          prop={'weight': DEFAULT_FONT_WEIGHT, 'size': _QF_LEGEND})
                # Total value on top
                totals = plot_data.sum(axis=1)
                ax.set_ylim(bottom=0, top=totals.max() * 1.15)
                pad = totals.max() * 0.01
                for i, val in enumerate(totals):
                    ax.text(i, val + pad, f"{val:.0f} ms", ha='center', va='bottom',
                            fontweight=DEFAULT_FONT_WEIGHT, fontsize=_QF_VALUE)
                # Per-segment annotations to the right
                for bar_idx in range(len(plot_data)):
                    _annotate_segments(ax, plot_data, bar_idx, fontsize=_QF_ANNOT)
                plt.savefig(out_path, dpi=600, bbox_inches='tight')
                plt.close(fig)

            # --- Plot 1: Detailed breakdown (Rel. Operators, VS, Data Movement, Index Movement, Other) ---
            p1 = get_query_data(df, row_idx, detailed=True)
            p1 = p1.dropna(how='all')
            p1 = p1[p1.sum(axis=1) > 0]
            if not p1.empty:
                out1 = get_plot_path(out_dir, benchmark, sf, system, index_desc, "plot_1", metric,
                                     f"detailed_highlevel_breakdown_{q}{_EXT}", per_query=True)
                title1 = f"Detailed Breakdown: {name}\n({system}, {index_desc}, SF={sf}{_K_LABEL}, {metric.upper()})"
                _quick_plot(p1, title1, out1)
                all_p.append(out1)

            # --- Plot 2: Operator breakdown (VectorSearch, Filter, Join, GroupBy, ...) ---
            p2 = get_query_operator_data(df, row_idx, coarse=False)
            p2 = p2.dropna(how='all')
            p2 = p2[p2.sum(axis=1) > 0]
            if not p2.empty:
                out2 = get_plot_path(out_dir, benchmark, sf, system, index_desc, "plot_2", metric,
                                     f"operator_breakdown_{q}{_EXT}", per_query=True)
                title2 = f"Operator Breakdown: {name}\n({system}, {index_desc}, SF={sf}{_K_LABEL}, {metric.upper()})"
                _quick_plot(p2, title2, out2, legend_title="Operators")
                all_p.append(out2)

        return all_p

    if want is None or 1 in want:
        # Plot 1: Standard (coarse 3-category: Operators/Data Movement/Other)
        if _PER_QUERY:
            all_p += plot_1_individual(df, targets, metric_ext, index_desc, sf, out_dir, metric, system, benchmark, detailed=False)
        all_p += plot_1_summary_grid(df, targets, metric_ext, index_desc, sf, out_dir, metric, system, benchmark, detailed=False)
        if PLOT_FIXED_Y_LIMITS:
            all_p += plot_1_summary_grid(df, targets, metric_ext, index_desc, sf, out_dir, metric, system, benchmark, detailed=False, fixed_ylim=fixed_ylim)
        all_p += plot_1_aggregate(df, targets, metric_ext, index_desc, sf, out_dir, metric, system, benchmark, detailed=False)
        # Plot 1: Detailed (Rel. Operators, Vector Search, Index Movement, Data Movement, Other)
        if _PER_QUERY:
            all_p += plot_1_individual(df, targets, metric_ext, index_desc, sf, out_dir, metric, system, benchmark, detailed=True)
        all_p += plot_1_summary_grid(df, targets, metric_ext, index_desc, sf, out_dir, metric, system, benchmark, detailed=True)
        if PLOT_FIXED_Y_LIMITS:
            all_p += plot_1_summary_grid(df, targets, metric_ext, index_desc, sf, out_dir, metric, system, benchmark, detailed=True, fixed_ylim=fixed_ylim)
        all_p += plot_1_aggregate(df, targets, metric_ext, index_desc, sf, out_dir, metric, system, benchmark, detailed=True)

    if want is None or 2 in want:
        # Plot 2: Per-Query operator breakdown
        if _PER_QUERY:
            all_p += plot_2_individual(df, targets, metric_ext, index_desc, sf, out_dir, metric, system, benchmark)
        # NOTE: Switching the other cases off for now.
        # all_p += plot_2_case_comparison(df, targets, metric_ext, index_desc, sf, out_dir, metric, system, benchmark)
        all_p += plot_2_summary_grid(df, targets, metric_ext, index_desc, sf, out_dir, metric, system, benchmark, coarse=False)
        # all_p += plot_2_summary_grid(df, targets, metric_ext, index_desc, sf, out_dir, metric, system, benchmark, coarse=True)
        # all_p += plot_2_summary_consolidated(df, targets, metric_ext, index_desc, sf, out_dir, metric, system, benchmark)
        # all_p += plot_2_aggregate(df, targets, metric_ext, index_desc, sf, out_dir, metric, system, benchmark)

    # "no cpu" variants: skip CPU case, keep only GPU cases (1, 2, 3)
    if want is None or 1 in want or 2 in want:
        saved_skip = _SKIP_CASES
        _SKIP_CASES = _SKIP_CASES | {"0: CPU-CPU-CPU"}
        if want is None or 1 in want:
            all_p += plot_1_summary_grid(df, targets, metric_ext, index_desc, sf, out_dir, metric, system, benchmark, detailed=False, suffix="_no_cpu")
            all_p += plot_1_summary_grid(df, targets, metric_ext, index_desc, sf, out_dir, metric, system, benchmark, detailed=True, suffix="_no_cpu")
            all_p += plot_1_aggregate(df, targets, metric_ext, index_desc, sf, out_dir, metric, system, benchmark, detailed=False, suffix="_no_cpu")
            all_p += plot_1_aggregate(df, targets, metric_ext, index_desc, sf, out_dir, metric, system, benchmark, detailed=True, suffix="_no_cpu")
        if want is None or 2 in want:
            all_p += plot_2_summary_grid(df, targets, metric_ext, index_desc, sf, out_dir, metric, system, benchmark, coarse=False, suffix="_no_cpu")
        _SKIP_CASES = saved_skip

    # Plot 3 & 4: Per-rep breakdowns (only when explicitly requested via --plot 3 or --plot 4)
    if per_rep_csv and os.path.exists(per_rep_csv):
        if want is not None and (3 in want or 4 in want):
            all_p += generate_per_rep_plots(per_rep_csv, index_desc, sf, out_dir, system, benchmark, plots=want)

    return all_p

def main():
    parser = argparse.ArgumentParser(description="Cleaned VSDS Results Plotter")
    parser.add_argument("--in_dir", default="./parse_caliper", help="Results directory")
    parser.add_argument("--sf", required=True, help="Scaling factor (supports comma-separated)")
    parser.add_argument("--csv", help="Specific CSV path override")
    parser.add_argument("--out_dir", default="plots", help="Output base")
    parser.add_argument("--system", default="dgx-spark-02", help="System prefix")
    parser.add_argument("--benchmark", default="vsds", help="Benchmark prefix (supports comma-separated)")
    parser.add_argument("--metric", nargs="+", default=["median"], help="Metrics to process")
    parser.add_argument("--no-values", action="store_true", help="Don't print values on top of bars")
    parser.add_argument("--no-std-bars", action="store_true", help="Don't print std bars for expected metrics")
    parser.add_argument("--fixed-y-limits", action="store_true", help="Fix y-axis limits to global max")
    parser.add_argument("--no-per-rep", action="store_true", help="Skip per-repetition plots (Plot 3 & 4)")
    parser.add_argument("--no-per-query", action="store_true", help="Skip individual per-query plots (Plot 1/2 per query)")
    parser.add_argument("--skip-cases", default="2",
                        help="Comma-separated case numbers to skip (default: '2' = GPU-CPU-GPU). E.g. '1,2'")
    parser.add_argument("--title", action="store_true", help="Show plot titles (default: off)")
    parser.add_argument("--plot", default=None, help="Comma-separated plot numbers to generate (e.g. 1,3). Default: all.")
    parser.add_argument("--k", default="100",
                        help="k value to select (default: 100). Matches files whose extra tag is exactly "
                             "'k_<value>'. Files with no extra tag (old-style) are always included.")
    parser.add_argument("--format", default="jpeg", choices=["jpeg", "pdf", "png", "svg"],
                        help="Output image format. Use 'pdf' for vector quality in LaTeX papers.")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: only per-query detailed plot_1 + detailed plot_2. No summaries/grids.")
    args = parser.parse_args()

    global SHOW_VALUES, PLOT_FIXED_Y_LIMITS, SHOW_STD_BARS, _K_LABEL, _EXT, _K_VALUE, SHOW_TITLE, _PER_QUERY, _SKIP_CASES, _QUICK_MODE
    SHOW_VALUES = not args.no_values
    PLOT_FIXED_Y_LIMITS = args.fixed_y_limits
    SHOW_STD_BARS = not args.no_std_bars
    _K_LABEL = f", K={args.k}"
    _K_VALUE = args.k
    _EXT = f".{args.format}"
    SHOW_TITLE  = args.title
    _PER_QUERY  = not args.no_per_query
    _QUICK_MODE = args.quick
    _SKIP_CASES = {f"{n}: " + {"0": "CPU-CPU-CPU", "1": "GPU-CPU-CPU",
                                "2": "GPU-CPU-GPU", "3": "GPU-GPU-GPU"}[n]
                   for n in args.skip_cases.split(",") if n.strip() in ("0", "1", "2", "3")}

    plots = {int(p.strip()) for p in args.plot.split(",")} if args.plot else None

    # Suffixes to match for the requested k value.
    k_suffixes = [f"_sf_{{}}_k_{args.k}.csv", "_sf_{{}}.csv"]

    all_done = []
    benchmarks = [b.strip() for b in args.benchmark.split(",")]
    sfs = [s.strip() for s in args.sf.split(",")]

    for benchmark in benchmarks:
        for sf in sfs:
            pre = f"{args.system}_{benchmark}_"
            suffixes = [t.format(sf) for t in k_suffixes]

            if args.csv:
                files = [(args.csv, "custom")]
            else:
                files = []
                for suf in suffixes:
                    for fpath in glob.glob(os.path.join(args.in_dir, benchmark, f"{pre}*{suf}")):
                        bname = os.path.basename(fpath)
                        if bname.startswith(pre) and bname.endswith(suf):
                            norm_idx = bname[len(pre):-len(suf)].replace("GPU,", "")
                            files.append((fpath, norm_idx))

            if not files:
                continue

            for fpath, norm_idx in files:
                print(f"-- Processing {fpath} (Index: {norm_idx}, Bench: {benchmark}, SF: {sf}) --")

                # Discover corresponding per-rep CSV
                per_rep_csv = None
                if not args.no_per_rep:
                    per_rep_path = os.path.join(args.in_dir, "per_rep", benchmark, os.path.basename(fpath))
                    if os.path.exists(per_rep_path):
                        per_rep_csv = per_rep_path

                for m in args.metric:
                    all_done.extend(generate_plots(fpath, norm_idx, sf, args.out_dir, m, args.system, benchmark, per_rep_csv=per_rep_csv, plots=plots))

    if not all_done:
        print(f"\nError: No files were processed. Check your --sf and --benchmark arguments.")
        sys.exit(1)

    print(f"\nDONE: {len(all_done)} plots saved.")

if __name__ == "__main__":
    main()
