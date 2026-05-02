"""
parse_varbatch.py — Parse vary-batch log files into CSVs.

Usage:
    cd results
    python parse_varbatch.py --sf 1 --system sgs-gpu05
"""

import re
import os
import math
import shutil
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Case mapping
# ---------------------------------------------------------------------------
CASE_MAPPING = {
    ("cpu", "cpu",        "cpu"):        "0: CPU-CPU-CPU",
    ("gpu", "cpu",        "cpu"):        "1: GPU-CPU-CPU",
    ("gpu", "cpu-pinned", "cpu-pinned"): "1P: GPU-CPU(P)-CPU(P)",
    ("gpu", "cpu",        "gpu"):        "2: GPU-CPU-GPU",
    ("gpu", "gpu",        "gpu"):        "3: GPU-GPU-GPU",
}

# Filename pattern — captures optional variant suffix after vary_batch_in_maxbench
# or bs1_fullsweep (e.g., _pre_hybrid, _low_sel, _pre_hybrid_low_sel, -ivf_h).
# q_{query}-i_{index}-d_{device}-s_{storage}-is_{index_storage}-sf_{sf}-{scenario}[_variant].log
# cpu-pinned listed before cpu so the longer alternation matches first.
FILENAME_RE = re.compile(
    r"q_(?P<query>.+?)-i_(?P<index>.+?)-d_(?P<device>cpu|gpu)"
    r"-s_(?P<storage>cpu-pinned|cpu|gpu)"
    r"-is_(?P<index_storage>cpu-pinned|cpu|gpu)"
    r"-sf_(?P<sf>[^-]+)"
    r"-(?P<scenario>vary_batch_in_maxbench|bs1_fullsweep)"
    r"(?:[_-](?P<variant>.+?))?\.log$"
)

# Recognized selectivity variant tags (aligned with parse_caliper.py VALID_EXTRA_KEYS pattern)
# Execution variants (pre_hybrid, post_hybrid): don't modify query name
# Selectivity variants: appended to query name for separate grouping in plots
SELECTIVITY_VARIANTS = {"low_sel", "high_sel"}

def extract_selectivity(variant_str):
    """Extract selectivity tag from a varbatch variant string (old format fallback).

    Examples:
      "low_sel"              -> "low_sel"
      "high_sel"             -> "high_sel"
      "pre_hybrid"           -> ""
      "pre_hybrid_low_sel"   -> "low_sel"
      "post_hybrid_high_sel" -> "high_sel"
      None                   -> ""
    """
    if not variant_str:
        return ""
    for sel in SELECTIVITY_VARIANTS:
        if sel in variant_str:
            return sel
    return ""

# Recognized extra-tag keys (same definitions as parse_caliper.py)
ALL_EXTRA_KEYS = {"k", "metric", "sel", "cagra", "ivf"}
INDEX_VARIANT_MAP = {"cagra": {"ch": "(C+H)", "c": "(C)"}, "ivf": {"h": "(H)"}}
QUERY_VARIANT_MAP = {"sel": {"low": "low_sel", "high": "high_sel"}}

def parse_variant_tags(variant_str):
    """Parse key-value tags from variant/extra_tag string (uses '-' as group separator).

    Examples:
      "cagra_ch"                    -> {'cagra': 'ch'}
      "sel_low"                     -> {'sel': 'low'}
      "pre_hybrid-cagra_ch-sel_low" -> {'cagra': 'ch', 'sel': 'low'}  ('pre' not recognized)
      "pre_hybrid"                  -> {}
      None                          -> {}
    """
    if not variant_str:
        return {}
    result = {}
    for group in variant_str.split("-"):
        parts = group.split("_")
        if len(parts) >= 2:
            key = parts[0].lower()
            if key in ALL_EXTRA_KEYS:
                result[key] = parts[1]
    return result

def apply_index_variant(base_index, parsed):
    """Rename index based on variant keys (e.g., cagra=ch -> Cagra(C+H), ivf=h -> IVF(H))."""
    import re
    for key, value_map in INDEX_VARIANT_MAP.items():
        val = parsed.get(key, "")
        if val in value_map:
            suffix = value_map[val]
            if key == "cagra" and "Cagra" in base_index:
                return base_index.replace("Cagra", f"Cagra{suffix}")
            elif key == "ivf" and "IVF" in base_index:
                return re.sub(r'(IVF\d+)', rf'\1{suffix}', base_index)
    return base_index

# Result CSV filename pattern — captures optional variant suffix (same scenarios as log)
# maximus_q_{query}-...-{scenario}[_variant]_batch_{N}_qstart_{Q}.csv
CSV_FILENAME_RE = re.compile(
    r"maximus_q_(?P<query>.+?)-i_(?P<index>.+?)-d_(?P<device>cpu|gpu)"
    r"-s_(?P<storage>cpu-pinned|cpu|gpu)"
    r"-is_(?P<index_storage>cpu-pinned|cpu|gpu)"
    r"-sf_(?P<sf>[^-]+)"
    r"-(?P<scenario>vary_batch_in_maxbench|bs1_fullsweep)"
    r"(?:[_-](?P<variant>.+?))?"
    r"_batch_(?P<batch_size>\d+)_qstart_(?P<qstart>\d+)\.csv$"
)

# Standard (non-varbatch) filename regex — optional params_tag suffix, no vary_batch_in_maxbench
FILENAME_RE_STANDARD = re.compile(
    r"q_(?P<query>.+?)-i_(?P<index>.+?)-d_(?P<device>cpu|gpu)"
    r"-s_(?P<storage>cpu-pinned|cpu|gpu)"
    r"-is_(?P<index_storage>cpu-pinned|cpu|gpu)"
    r"-sf_(?P<sf>[^-]+?)(?:-(?P<params_tag>.+?))?\.log$"
)

# Standard CSV regex — no batch_size/qstart suffix
CSV_FILENAME_RE_STANDARD = re.compile(
    r"maximus_q_(?P<query>.+?)-i_(?P<index>.+?)-d_(?P<device>cpu|gpu)"
    r"-s_(?P<storage>cpu-pinned|cpu|gpu)"
    r"-is_(?P<index_storage>cpu-pinned|cpu|gpu)"
    r"-sf_(?P<sf>[^-]+?)(?:-(?P<params_tag>.+?))?\.csv$"
)

# Timing line in standard logs
MAXIMUS_TIMINGS_RE = re.compile(r"- MAXIMUS TIMINGS \[ms\]:\s+([\d,\s]+)")

# [VSDS Parameters] block entry pattern
VSDS_PARAMS_RE = re.compile(r"---> ([\w_]+(?:\s+\(if [^)]+\))?):\s+(.+)")

# Per-table primary key column names used in result CSVs
TABLE_QUERY_COL = {"reviews": "rv_reviewkey_queries", "images": "i_imagekey_queries"}
TABLE_NN_COL    = {"reviews": "rv_reviewkey",         "images": "i_imagekey"}

# Number of queries (= batch size) used as ground truth per scale factor.
# The GT CSV must start at qstart=0.
# Extend this dict when new scale factors are added.
SF_RECALL_BATCH = {
    "1":    10000,
    "0.01": 1000,
}


# ---------------------------------------------------------------------------
# Caliper region parsing helpers (mirrored from parse_caliper.py)
# ---------------------------------------------------------------------------

CALIPER_REGIONS = [
    "Executor::execute", "OPERATORS", "_CUDF_", "_ACERO_", "_NATIVE_", "_FAISS_",
    "DataTransformation", "CPU->GPU", "GPU->CPU", "CPU->CPU", "GPU->GPU", "index_movement",
]

_CALIPER_BS_RE     = re.compile(r"^batch_size_(\d+)\s")
_CALIPER_QBATCH_RE = re.compile(r"^(\w+)_batch_(\d+)_qstart_(\d+)\s")
_CALIPER_REP_RE    = re.compile(r"^Repetition_(\d+)\s")


def _caliper_compute_stats(values_list):
    if not values_list:
        return {"mean": 0.0, "min": 0.0, "median": 0.0, "max": 0.0, "std": 0.0}
    vals = np.array(values_list, dtype=float)
    return {
        "mean":   float(np.mean(vals)),
        "min":    float(np.min(vals)),
        "median": float(np.median(vals)),
        "max":    float(np.max(vals)),
        "std":    float(np.std(vals)) if len(vals) > 1 else 0.0,
    }


def _caliper_classify_region(region_key, val):
    """Classify a Caliper region name+value into timing category increments."""
    inc = defaultdict(float)
    if "Executor::execute" in region_key:
        inc["Total"]     += val
        inc["_overhead"] += val
    if "CPU->GPU" in region_key or "GPU->CPU" in region_key:
        inc["Data Transfers"] += val
    if "OPERATORS" in region_key:
        inc["Operators"]  += val
        inc["_overhead"]  -= val
    if "CPU->CPU" in region_key or "GPU->GPU" in region_key:
        inc["Data Conversions"] += val
    if "index_movement" in region_key:
        inc["IndexMovement"] += val
    if "DISTINCT" in region_key:
        inc["Distinct"]   += val;  inc["_ops_time"] += val
    if "SCATTER" in region_key:
        inc["Scatter"] += val; inc["_ops_time"] += val
    if "GATHER" in region_key:
        inc["Gather"]  += val; inc["_ops_time"] += val
    if "FILTER" in region_key:
        inc["Filter"]     += val;  inc["_ops_time"] += val
    if "PROJECT" in region_key:
        inc["Project"]    += val;  inc["_ops_time"] += val
    if "JOIN" in region_key:
        inc["Join"]       += val;  inc["_ops_time"] += val
    if "VECTOR_SEARCH" in region_key:
        inc["VectorSearch"] += val; inc["_ops_time"] += val
    if "GROUP_BY" in region_key:
        inc["GroupBy"]    += val;  inc["_ops_time"] += val
    if "ORDER_BY" in region_key:
        inc["OrderBy"]    += val;  inc["_ops_time"] += val
    if "LIMIT" in region_key:
        inc["Limit"]      += val;  inc["_ops_time"] += val
    if "TAKE" in region_key:
        inc["Take"]       += val;  inc["_ops_time"] += val
    if "LOCAL_BROADCAST" in region_key:
        inc["LocalBroadcast"] += val; inc["_ops_time"] += val
    return inc


def parse_caliper_regions_varbatch(filepath, incl_rep0=False):
    """Parse Caliper region timings from a varbatch log file.

    Returns: dict[(batch_size, query, qstart)] -> dict[region_name -> list[float_ms]]
    Each float_ms value is the inclusive time in ms for one rep (rep 0 skipped unless
    incl_rep0=True).
    """
    with open(filepath) as f:
        lines = f.readlines()

    # Find Caliper runtime-report header (e.g. "Path   Time (E)   Time (I) ...")
    caliper_start = None
    for i, line in enumerate(lines):
        if re.match(r"\s*Path\s+Time", line):
            caliper_start = i + 1
            break
    if caliper_start is None:
        return {}

    result = defaultdict(lambda: defaultdict(list))

    cur_bs     = None
    cur_query  = None
    cur_qstart = None
    cur_rep    = None
    in_execute = False

    bs_indent    = None
    query_indent = None
    rep_indent   = None

    for line in lines[caliper_start:]:
        raw = line.rstrip("\n")
        if not raw.strip():
            continue

        indent   = len(raw) - len(raw.lstrip())
        stripped = raw.strip()

        # Context resets based on indentation (outermost first)
        if bs_indent is not None and indent <= bs_indent:
            cur_bs = None;     bs_indent    = None
            cur_query = None;  query_indent = None
            cur_rep = None;    rep_indent   = None
            in_execute = False
        elif query_indent is not None and indent <= query_indent:
            cur_query = None;  query_indent = None
            cur_rep = None;    rep_indent   = None
            in_execute = False
        elif rep_indent is not None and indent <= rep_indent:
            cur_rep = None;    rep_indent   = None
            in_execute = False

        # Match batch_size_N
        m = _CALIPER_BS_RE.match(stripped)
        if m:
            cur_bs    = int(m.group(1))
            bs_indent = indent
            continue

        # Match query batch tag (only inside batch_size)
        if cur_bs is not None:
            m = _CALIPER_QBATCH_RE.match(stripped)
            if m:
                cur_query    = m.group(1)
                cur_qstart   = int(m.group(3))
                query_indent = indent
                continue

        # Match Repetition_X (only inside query batch)
        if cur_query is not None:
            m = _CALIPER_REP_RE.match(stripped)
            if m:
                cur_rep    = int(m.group(1))
                rep_indent = indent
                in_execute = False
                continue

        # Need full context to proceed
        if cur_bs is None or cur_query is None or cur_rep is None:
            continue
        if not incl_rep0 and cur_rep == 0:
            continue

        # Mark start of Executor::execute block
        if not in_execute:
            if "Executor::execute" in stripped:
                in_execute = True
            else:
                continue

        # Parse any matching region on this line
        for region in CALIPER_REGIONS:
            if region in stripped:
                cols = stripped.replace("'", "").split()
                if len(cols) >= 3:
                    try:
                        val_ms = float(cols[2]) * 1000.0  # inclusive seconds → ms
                        key = (cur_bs, cur_query, cur_qstart)
                        result[key][cols[0]].append(val_ms)
                    except (ValueError, IndexError):
                        pass
                break

    return result


_BREAKDOWN_COLS = [
    "Total_ms", "Operators_ms", "VectorSearch_ms", "Filter_ms", "Project_ms",
    "Join_ms", "GroupBy_ms", "OrderBy_ms", "Limit_ms", "Take_ms",
    "LocalBroadcast_ms", "Scatter_ms", "Gather_ms", "Distinct_ms",
    "DataTransfers_ms", "DataConversions_ms", "IndexMovement_ms", "Other_ms",
]


def aggregate_operator_breakdown(caliper_data, timing_rows, query, case_label,
                                 params_tag, search_param, caliper_query=None):
    """Aggregate Caliper region data into operator breakdown rows.

    caliper_data: dict[(batch_size, query, qstart)] -> dict[region -> list[ms]]
    timing_rows: list of timing dicts (for sanity check).
    caliper_query: query name as it appears in the caliper tree (before selectivity
        rename). If None, defaults to query.
    Returns list of flat row dicts with columns defined in _BREAKDOWN_COLS.
    """
    caliper_query = caliper_query or query

    # Build lookup: batch_size -> mean_ms from timing lines (for sanity check)
    timing_by_bs = {}
    for t in timing_rows:
        if t.get("batch_size") is not None:
            timing_by_bs[int(t["batch_size"])] = t.get("mean_ms", 0.0)

    # Group caliper data by batch_size (aggregating across qstarts)
    # For each batch_size, pool all rep values per region
    by_bs = defaultdict(lambda: defaultdict(list))
    for (bs, q, qstart), region_map in caliper_data.items():
        if q != caliper_query:
            continue
        for region_name, vals in region_map.items():
            by_bs[bs][region_name].extend(vals)

    rows = []
    for bs in sorted(by_bs.keys()):
        region_map = by_bs[bs]

        for stat in ["mean", "min", "median", "max", "std"]:
            qt = defaultdict(float)
            for region_key, vals in region_map.items():
                s = _caliper_compute_stats(vals)
                val = s[stat]
                for k, v in _caliper_classify_region(region_key, val).items():
                    qt[k] += v

            overhead    = qt.pop("_overhead", 0.0)
            ops_time    = qt.pop("_ops_time", 0.0)
            qt["Operators"] = ops_time
            qt["Total"]     = ops_time + overhead + qt["Data Transfers"] + qt["Data Conversions"]
            qt["Other"]     = overhead + qt["Data Conversions"]

            # Sanity check (mean only, warn if >5% deviation from wall-clock timing)
            if stat == "mean" and bs in timing_by_bs:
                wall_ms = timing_by_bs[bs]
                if wall_ms > 0 and abs(qt["Total"] - wall_ms) / wall_ms > 0.05:
                    print(f"  [WARNING] Caliper total {qt['Total']:.3f} ms vs "
                          f"timing {wall_ms:.3f} ms (>{5}% diff) for "
                          f"query={query} case={case_label} batch_size={bs}")

            row = {
                "query":         query,
                "case":          case_label,
                "batch_size":    bs,
                "params_tag":    params_tag,
                "search_param":  search_param,
                "stat":          stat,
                "Total_ms":           qt.get("Total", 0.0),
                "Operators_ms":       qt.get("Operators", 0.0),
                "VectorSearch_ms":    qt.get("VectorSearch", 0.0),
                "Filter_ms":          qt.get("Filter", 0.0),
                "Project_ms":         qt.get("Project", 0.0),
                "Join_ms":            qt.get("Join", 0.0),
                "GroupBy_ms":         qt.get("GroupBy", 0.0),
                "OrderBy_ms":         qt.get("OrderBy", 0.0),
                "Limit_ms":           qt.get("Limit", 0.0),
                "Take_ms":            qt.get("Take", 0.0),
                "LocalBroadcast_ms":  qt.get("LocalBroadcast", 0.0),
                "Scatter_ms":         qt.get("Scatter", 0.0),
                "Gather_ms":          qt.get("Gather", 0.0),
                "Distinct_ms":        qt.get("Distinct", 0.0),
                "DataTransfers_ms":   qt.get("Data Transfers", 0.0),
                "DataConversions_ms": qt.get("Data Conversions", 0.0),
                "IndexMovement_ms":   qt.get("IndexMovement", 0.0),
                "Other_ms":           qt.get("Other", 0.0),
            }
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Recall helpers
# ---------------------------------------------------------------------------

def _table_of(query):
    if "reviews" in query:
        return "reviews"
    if "images" in query:
        return "images"
    return None


def _clean_csv(df):
    """Strip type annotations (e.g. '(int64)') from column names and trim whitespace."""
    df.columns = df.columns.str.replace(r'\([^)]*\)', '', regex=True).str.strip()
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].str.strip()
    return df


def _mean_recall_at_k(gt_df, ret_df, query_col, nn_col, k):
    """Mean recall@k across all queries in gt_df.

    Groups both DataFrames by query_col, takes the first k rows per query
    (assumes rows are already sorted by distance), then computes recall as
    |intersection| / |ground_truth| per query and returns the average.
    Works correctly even when gt and ret cover different subsets of queries
    (only queries present in gt are evaluated).
    """
    gt_df  = gt_df.groupby(query_col, sort=False).head(k)
    ret_df = ret_df.groupby(query_col, sort=False).head(k)

    gt_sets  = gt_df.groupby(query_col)[nn_col].apply(set)
    ret_sets = ret_df.groupby(query_col)[nn_col].apply(set)

    recalls = []
    for qid, truth in gt_sets.items():
        if not truth:
            continue
        retrieved = ret_sets.get(qid, set())
        recalls.append(len(truth & retrieved) / len(truth))

    return float(np.mean(recalls)) if recalls else float('nan')


PRE_VARIANT_TO_CORE = {
    "pre_reviews_partitioned": "pre_reviews",
    "pre_images_partitioned":  "pre_images",
    "pre_reviews_hybrid":      "pre_reviews",
    "pre_images_hybrid":       "pre_images",
}

POST_TO_PRE = {
    "post_reviews":             "pre_reviews",
    "post_images":              "pre_images",
    "post_reviews_hybrid":      "pre_reviews",
    "post_images_hybrid":       "pre_images",
    "post_reviews_partitioned": "pre_reviews",
    "post_images_partitioned":  "pre_images",
}

def _resolve_post_to_pre(query):
    """Map post_* query to its pre_* GT equivalent, handling selectivity suffixes.

    Examples:
      "post_images"          -> "pre_images"
      "post_images_low_sel"  -> "pre_images_low_sel"
      "post_images_hybrid_high_sel" -> "pre_images_high_sel"
    """
    for post_prefix, pre_prefix in POST_TO_PRE.items():
        if query.startswith(post_prefix):
            suffix = query[len(post_prefix):]  # e.g., "_low_sel" or ""
            return pre_prefix + suffix
    return None


def _cross_check(label, path_a, path_b, query_col, nn_col, k, threshold):
    """Compute recall@k of path_b against path_a and print CHECK/WARN."""
    try:
        r = _mean_recall_at_k(
            _clean_csv(pd.read_csv(path_a)),
            _clean_csv(pd.read_csv(path_b)),
            query_col, nn_col, k,
        )
        if r < threshold:
            print(f"  [WARN]   Cross-check FAILED {label}: recall@{k} = {r:.4f} "
                  f"(threshold {threshold})")
        else:
            print(f"  [CHECK]  Cross-check OK    {label}: recall@{k} = {r:.4f}")
    except Exception as e:
        print(f"  [WARN]   Cross-check error  {label}: {e}")


def parse_log_file_standard(filepath):
    """Parse a standard (non-varbatch) log returning one timing dict (no batch_size)."""
    with open(filepath) as f:
        content = f.read()
    m = MAXIMUS_TIMINGS_RE.search(content)
    if not m:
        return []
    raw = [float(x.strip()) for x in m.group(1).split(',') if x.strip()]
    times = raw[1:] if len(raw) > 1 else raw   # skip rep 0 (warmup)
    if not times:
        return []
    arr = np.array(times)
    n   = len(arr)
    valid = arr[arr > 0]
    if len(valid) < n:
        print(f"[WARN] {n - len(valid)} zero-timing rep(s) dropped in {filepath.name}")
    if not len(valid):
        print(f"[WARN] All reps zero in {filepath.name}")
        return []
    return [{"mean_ms":   float(np.mean(arr)),
             "min_ms":    float(np.min(arr)),
             "median_ms": float(np.median(arr)),
             "max_ms":    float(np.max(arr)),
             "std_ms":    float(np.std(arr)) if n > 1 else 0.0}]
    # No QPS: query count unknown at parse time (not in log file).


def parse_vsds_params(filepath):
    """Extract [VSDS Parameters] block → dict of param name → value string."""
    params = {}
    in_block = False
    with open(filepath) as f:
        for line in f:
            if line.strip() == "[VSDS Parameters]":
                in_block = True
                continue
            if in_block:
                m = VSDS_PARAMS_RE.match(line)
                if m:
                    params[m.group(1).strip()] = m.group(2).strip()
                elif line.strip() == "":
                    break   # blank line ends the block
    return params


def get_index_search_param(index, device, query, vsds_params):
    """Return human-readable search_param string for the index type."""
    base_index = index.removeprefix("GPU,")
    parts = []
    if base_index.startswith("IVF"):
        nprobe = vsds_params.get("ivf_nprobe", "")
        if nprobe:
            parts.append(f"nprobe={nprobe}")
    elif base_index.startswith("Cagra") and device == "gpu":
        itopk = vsds_params.get("cagra_itopksize", "")
        sw    = vsds_params.get("cagra_searchwidth", "")
        if itopk:
            parts.append(f"itopk={itopk}")
        if sw:
            parts.append(f"sw={sw}")
    elif base_index.startswith("Cagra") and device == "cpu":
        # CPU Cagra → HNSWCagra internally → uses efSearch
        efsearch = vsds_params.get("hnsw_efsearch", "")
        if efsearch:
            parts.append(f"efsearch={efsearch}")
    elif base_index.startswith("HNSW"):
        efsearch = vsds_params.get("hnsw_efsearch", "")
        if efsearch:
            parts.append(f"efsearch={efsearch}")
    if "post" in query:  # catches post_*, post_*_partitioned, post_*_hybrid
        ksearch = vsds_params.get("postfilter_ksearch", "")
        if ksearch:
            parts.append(f"ksearch={ksearch}")
    return ",".join(parts)


def _iter_recall_csvs(scan_dir, system, sf, recall_batch_size):
    """Yield (query, index, case_label, table, filepath, base_index, is_cpu, params_tag)
    for all matching CSV files in scan_dir/{cpu,gpu}-system/sf_sf/csv/.

    Varbatch CSVs: filtered to batch_size==recall_batch_size, qstart==0.
    Standard CSVs: all files included (no batch_size filter); params_tag carries variant info.
    """
    for sys_folder in [f"cpu-{system}", f"gpu-{system}"]:
        folder = scan_dir / sys_folder / f"sf_{sf}" / "csv"
        if not folder.is_dir():
            continue
        for fname in sorted(os.listdir(folder)):
            params_tag = None
            m = CSV_FILENAME_RE.match(fname)             # try varbatch first
            if m and m.group("sf") == sf:
                if int(m.group("batch_size")) != recall_batch_size:
                    continue
                if int(m.group("qstart")) != 0:
                    continue
            else:
                m = CSV_FILENAME_RE_STANDARD.match(fname)  # fallback: standard
                if not m or m.group("sf") != sf:
                    continue
                params_tag = m.group("params_tag")
            query      = m.group("query")
            index      = m.group("index")
            case_key   = (m.group("device"), m.group("storage"), m.group("index_storage"))
            case_label = CASE_MAPPING.get(case_key)
            if case_label is None:
                continue
            table = _table_of(query)
            if table is None:
                continue
            yield (query, index, case_label, table,
                   folder / fname,
                   index.removeprefix("GPU,"),
                   case_label == "0: CPU-CPU-CPU",
                   params_tag)


def _auto_copy_gt(base_dir, gt_dir, system, sf, recall_batch_size):
    """Copy GT files (enn_*, pre_*, ann_*,Flat at qstart=0) from base_dir to gt_dir
    if not already present there. Called before the GT scan pass."""
    for (query, index, case_label, table, filepath,
         base_index, is_cpu, params_tag) in \
            _iter_recall_csvs(base_dir, system, sf, recall_batch_size):
        is_gt = (query.startswith("enn_") or
                 query in POST_TO_PRE.values() or
                 query in PRE_VARIANT_TO_CORE or
                 (query in ("ann_reviews", "ann_images") and base_index == "Flat"))
        if not is_gt:
            continue
        rel  = filepath.relative_to(base_dir)
        dest = gt_dir / rel
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(filepath, dest)
            print(f"  [GT-COPY] {filepath.name}")


def compute_all_recalls(base_dir, system, sf, recall_batch_size=10000, k=100,
                        cross_check_warn_threshold=0.99, include_all_gpu_cases=False,
                        gt_dir=None):
    """Compute recall@k for all ANN result CSVs at batch=recall_batch_size.

    Ground truth (GT) is kept separate for CPU and GPU cases because
    enn_* on GPU is known to be unreliable:

      CPU GT (per table, best available):
        1. enn_*      on CPU  (exhaustive, exact)
        2. ann_*,Flat on CPU  (brute-force Flat, also exact; only ann_reviews/ann_images)

      GPU GT (per table, best available):
        1. ann_*,Flat on GPU case 1 (GPU-CPU-CPU)
        2. ann_*,Flat on GPU case 2 (GPU-CPU-GPU)
        3. ann_*,Flat on GPU case 3 (GPU-GPU-GPU)
        (enn_* on GPU is intentionally excluded)

    Cross-checks performed (warn if recall < threshold):
      - ann_*,Flat (any case) vs enn_* (CPU GT) — verifies GPU Flat == exact
      - pre_*_partitioned / pre_*_hybrid vs core pre_* — verifies execution variants agree

    ANN indexes on CPU case 0 are evaluated against CPU GT.
    ANN indexes on GPU cases 1-3 are evaluated against GPU GT.
    post_* queries are evaluated against the corresponding pre_* results (same case + index).

    Returns list of dicts: {query, case, index_name, mean_recall, recall_k}.
    """
    gt_cpu_cands   = {}   # table -> [(priority, filepath), ...]
    gt_gpu_cands   = {}   # table -> [(priority, filepath), ...]
    ann_flat_entries = [] # (case_label, table, base_index, filepath) at qstart=0
    ann_entries    = []   # (query, case_label, base_index, filepath, params_tag)
    post_entries   = []   # (query, case_label, base_index, filepath, params_tag)
    pre_gt         = {}   # (case_label, query_name, base_index) -> filepath
    pre_variant_gt = {}   # (case_label, query_name, base_index) -> filepath
    has_enn_cpu    = set()

    GPU_FLAT_GT_PRIORITY = {"1: GPU-CPU-CPU": 0, "2: GPU-CPU-GPU": 1, "3: GPU-GPU-GPU": 2}

    effective_gt_dir = gt_dir if (gt_dir and gt_dir != base_dir) else base_dir
    if gt_dir and gt_dir != base_dir:
        _auto_copy_gt(base_dir, gt_dir, system, sf, recall_batch_size)

    # Pass 1 — GT scan (over effective_gt_dir)
    for (query, index, case_label, table, filepath,
         base_index, is_cpu, params_tag) in \
            _iter_recall_csvs(effective_gt_dir, system, sf, recall_batch_size):

        if query in POST_TO_PRE.values():
            pre_gt[(case_label, query, base_index)] = filepath
        elif query in PRE_VARIANT_TO_CORE:
            pre_variant_gt[(case_label, query, base_index)] = filepath
        elif query.startswith("enn_") and is_cpu:
            gt_cpu_cands.setdefault(table, []).append((0, filepath))
            has_enn_cpu.add(table)
        elif query in ("ann_reviews", "ann_images") and base_index == "Flat":
            ann_flat_entries.append((case_label, table, base_index, filepath))
            if is_cpu:
                gt_cpu_cands.setdefault(table, []).append((1, filepath))
            else:
                prio = GPU_FLAT_GT_PRIORITY.get(case_label, 99)
                gt_gpu_cands.setdefault(table, []).append((prio, filepath))
        # enn_* GPU: intentionally skipped — known to be unreliable
        # POST and ANN non-Flat entries are handled in Pass 2

    # Pass 2 — Results scan (over base_dir)
    for (query, index, case_label, table, filepath,
         base_index, is_cpu, params_tag) in \
            _iter_recall_csvs(base_dir, system, sf, recall_batch_size):

        if _resolve_post_to_pre(query) is not None:
            post_entries.append((query, case_label, base_index, filepath, params_tag))
        elif (query not in POST_TO_PRE.values() and
              query not in PRE_VARIANT_TO_CORE and
              not query.startswith("enn_") and
              not (query in ("ann_reviews", "ann_images") and base_index == "Flat")):
            ann_entries.append((query, case_label, base_index, filepath, params_tag))

    # Pick best GT per table
    gt_cpu = {}
    for table, cands in gt_cpu_cands.items():
        cands.sort(key=lambda x: x[0])
        gt_cpu[table] = cands[0][1]
        print(f"  [GT-CPU] {table:8s} → {cands[0][1].name}")

    gt_gpu = {}
    for table, cands in gt_gpu_cands.items():
        cands.sort(key=lambda x: x[0])
        gt_gpu[table] = cands[0][1]
        print(f"  [GT-GPU] {table:8s} → {cands[0][1].name}")

    for table in gt_cpu:
        if table not in has_enn_cpu:
            print(f"  [WARN]   No enn_* GT found for '{table}' (CPU); "
                  f"falling back to ann_*,Flat as GT: {gt_cpu[table].name}")

    # Cases included in cross-checks and recall evaluation
    EVAL_CASES = {
        "0: CPU-CPU-CPU", "1: GPU-CPU-CPU", "2: GPU-CPU-GPU", "3: GPU-GPU-GPU",
    } if include_all_gpu_cases else {
        "0: CPU-CPU-CPU", "3: GPU-GPU-GPU",
    }

    # Cross-check 1: ann_*,Flat (per case) vs enn_* (CPU GT)
    # Covers both CPU sanity (ann_Flat == enn on CPU) and GPU correctness (GPU Flat == CPU ENN).
    for case_label, table, base_index, filepath in sorted(ann_flat_entries):
        if case_label not in EVAL_CASES:
            continue
        if table not in gt_cpu or filepath == gt_cpu[table]:
            continue  # no enn GT, or this file IS the CPU GT (enn absent)
        query_col, nn_col = TABLE_QUERY_COL[table], TABLE_NN_COL[table]
        _cross_check(f"ann,Flat({case_label}) vs enn ({table})",
                     gt_cpu[table], filepath, query_col, nn_col, k, cross_check_warn_threshold)

    # Cross-check 2: pre_*_partitioned / pre_*_hybrid vs core pre_*
    # Hybrid variants always cross-checked regardless of EVAL_CASES
    for (case_label, variant_query, base_index), variant_path in sorted(pre_variant_gt.items()):
        is_hybrid = variant_query.endswith("_hybrid")
        if not is_hybrid and case_label not in EVAL_CASES:
            continue
        core_key = (case_label, PRE_VARIANT_TO_CORE[variant_query], base_index)
        if core_key not in pre_gt:
            print(f"  [WARN]   No core pre_* GT for '{variant_query}' "
                  f"(case={case_label}, index={base_index}); skipping cross-check")
            continue
        table = _table_of(variant_query)
        query_col, nn_col = TABLE_QUERY_COL[table], TABLE_NN_COL[table]
        _cross_check(f"'{variant_query}' vs '{PRE_VARIANT_TO_CORE[variant_query]}' "
                     f"(case={case_label}, index={base_index})",
                     pre_gt[core_key], variant_path, query_col, nn_col, k, cross_check_warn_threshold)

    # Filter ann_entries to EVAL_CASES (defined above)
    skipped = [(q, c, i, f, p) for q, c, i, f, p in ann_entries if c not in EVAL_CASES]
    ann_entries = [(q, c, i, f, p) for q, c, i, f, p in ann_entries if c in EVAL_CASES]
    if skipped:
        print(f"  [SKIP]   {len(skipped)} entries skipped (cases 1 & 2; use --all-gpu-cases to include)")

    # Evaluate each ANN index
    recall_rows = []
    for query, case_label, base_index, filepath, params_tag in sorted(ann_entries):
        table     = _table_of(query)
        is_cpu    = (case_label == "0: CPU-CPU-CPU")
        gt_files  = gt_cpu if is_cpu else gt_gpu

        if table not in gt_files:
            fallback = gt_cpu if not is_cpu else {}
            if table in fallback:
                print(f"  [WARN]   No GPU GT for '{table}', falling back to CPU GT "
                      f"for {filepath.name}")
                gt_files = fallback
            else:
                print(f"  [WARN]   No ground truth for '{table}', skipping {filepath.name}")
                continue

        query_col = TABLE_QUERY_COL[table]
        nn_col    = TABLE_NN_COL[table]

        try:
            gt_df  = _clean_csv(pd.read_csv(gt_files[table]))
            ret_df = _clean_csv(pd.read_csv(filepath))
            r = _mean_recall_at_k(gt_df, ret_df, query_col, nn_col, k)
            recall_rows.append({
                "query":       query,
                "case":        case_label,
                "index_name":  base_index,
                "params_tag":  params_tag,
                "mean_recall": round(r, 6),
                "recall_k":    k,
                "gt_filepath": str(gt_files[table]),
                "filepath":    str(filepath),
            })
            print(f"  [RECALL] {query:<20} | {base_index:<25} | {case_label}: "
                  f"recall@{k} = {r:.4f}")
        except Exception as e:
            print(f"  [ERROR]  Recall failed for {filepath.name}: {e}")

    # Evaluate post_* entries against matching pre_* GT
    # Hybrid variants always evaluated regardless of EVAL_CASES
    for query, case_label, base_index, filepath, params_tag in sorted(post_entries):
        is_hybrid = query.endswith("_hybrid")
        if not is_hybrid and case_label not in EVAL_CASES:
            continue
        pre_query = _resolve_post_to_pre(query)
        if pre_query is None:
            print(f"  [WARN]   No POST_TO_PRE mapping for '{query}'; skipping recall")
            continue
        # pre_* always runs with Flat index on CPU — ignore the post query's case/index
        key = ("0: CPU-CPU-CPU", pre_query, "Flat")
        if key not in pre_gt:
            print(f"  [WARN]   No pre_* GT for post query '{query}' "
                  f"(case={case_label}, index={base_index}); skipping {filepath.name}")
            continue
        table     = _table_of(query)
        query_col = TABLE_QUERY_COL[table]
        nn_col    = TABLE_NN_COL[table]
        try:
            gt_df  = _clean_csv(pd.read_csv(pre_gt[key]))
            ret_df = _clean_csv(pd.read_csv(filepath))
            r = _mean_recall_at_k(gt_df, ret_df, query_col, nn_col, k)
            recall_rows.append({
                "query":       query,
                "case":        case_label,
                "index_name":  base_index,
                "params_tag":  params_tag,
                "mean_recall": round(r, 6),
                "recall_k":    k,
                "gt_filepath": str(pre_gt[key]),
                "filepath":    str(filepath),
            })
            print(f"  [RECALL] {query:<20} | {base_index:<25} | {case_label}: "
                  f"recall@{k} = {r:.4f}  [GT=pre]")
        except Exception as e:
            print(f"  [ERROR]  Recall failed for {filepath.name}: {e}")

    return recall_rows


def parse_log_file_bs1(filepath):
    """bs1_fullsweep-specific: one row per (batch_size, qstart), preserving per-query tracking.

    Reads Caliper `Executor::execute` (pure operator pipeline time, excludes db->schedule +
    final barrier) rather than the chrono printed line. Rep 0 skipped as warmup.
    """
    caliper_data = parse_caliper_regions_varbatch(filepath, incl_rep0=False)
    results = []
    for (batch_size, _query, qstart), region_map in caliper_data.items():
        vals = region_map.get("Executor::execute", [])
        if not vals:
            continue
        arr = np.array(vals)
        n   = len(arr)
        results.append({
            "batch_size": batch_size,
            "qstart":     qstart,
            "mean_ms":    float(np.mean(arr)),
            "min_ms":     float(np.min(arr)),
            "median_ms":  float(np.median(arr)),
            "max_ms":     float(np.max(arr)),
            "std_ms":     float(np.std(arr)) if n > 1 else 0.0,
        })
    return results


def parse_setup_index_movement_ms(filepath):
    """Extract inclusive time (ms) of the CASE 6 setup_index_movement Caliper region.

    The region is emitted exactly once per run, outside the vary_batch loop, so it appears
    as a top-level entry in the runtime-report table. Returns None if not found (i.e. run
    was not CASE 6 / --case6_persist_gpu_index was off).
    """
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith("setup_index_movement"):
                continue
            parts = stripped.split()
            # Columns: Path  Time(E)  Time(I)  Time%(E)  Time%(I)
            if len(parts) >= 3:
                try:
                    return float(parts[2]) * 1000.0  # inclusive seconds → ms
                except ValueError:
                    return None
    return None


def parse_log_file_caliper(filepath, query_name):
    """Per-batch timing + QPS stats from Caliper `Executor::execute` (inclusive).

    Replaces the old chrono-based parser — chrono wrapped `db->schedule + execute + barrier`
    and carried ~1-3 ms framework overhead that dominated fast indexes at bs=1. Executor::execute
    is the pure operator-pipeline time. Emits one row per (batch_size, qstart) matching the
    shape of parse_log_file_standard / _bs1 so downstream code is unaffected. Rep 0 skipped.
    """
    caliper_data = parse_caliper_regions_varbatch(filepath, incl_rep0=False)
    results = []
    for (batch_size, q, qstart), region_map in caliper_data.items():
        if q != query_name:
            continue
        vals = region_map.get("Executor::execute", [])
        if not vals:
            continue
        arr = np.array(vals)
        n = len(arr)

        valid = arr[arr > 0]
        if len(valid) < n:
            print(f"[WARN] {n - len(valid)} zero-timing rep(s) dropped for "
                  f"batch_size={batch_size} qstart={qstart} in {filepath.name}")
        if len(valid) == 0:
            print(f"[WARN] All reps are zero for batch_size={batch_size} qstart={qstart}, "
                  f"skipping QPS")
            qps_arr = np.array([0.0])
        else:
            qps_arr = batch_size * 1000.0 / valid

        results.append({
            "batch_size": batch_size,
            "mean_ms":    float(np.mean(arr)),
            "min_ms":     float(np.min(arr)),
            "median_ms":  float(np.median(arr)),
            "max_ms":     float(np.max(arr)),
            "std_ms":     float(np.std(arr)) if n > 1 else 0.0,
            "mean_qps":   float(np.mean(qps_arr)),
            "min_qps":    float(np.min(qps_arr)),
            "median_qps": float(np.median(qps_arr)),
            "max_qps":    float(np.max(qps_arr)),
            "std_qps":    float(np.std(qps_arr)) if len(qps_arr) > 1 else 0.0,
        })
    return sorted(results, key=lambda r: (r["batch_size"], ))


def main():
    parser = argparse.ArgumentParser(description="Parse vary-batch logs into CSVs.")
    parser.add_argument("--sf", type=str, required=True, help="Scale factor (e.g. 1)")
    parser.add_argument("--system", type=str, default="sgs-gpu05", help="System name")
    parser.add_argument("--all-gpu-cases", action="store_true",
                        help="Include cases 1 & 2 (GPU-CPU-CPU, GPU-CPU-GPU) in recall. "
                             "By default only case 0 (CPU) and case 3 (GPU-GPU-GPU) are evaluated.")
    parser.add_argument("--result_dir", type=str, default="other-vary-batch-in-maxbench",
                        help="Experiment directory (relative to results/). "
                             "Default: other-vary-batch-in-maxbench")
    parser.add_argument("--gt_dir", type=str, default="ground-truth",
                        help="GT directory (relative to results/). "
                             "Auto-populated from result_dir on first run. Default: ground-truth")
    parser.add_argument("--parse_caliper", action="store_true",
                        help="Parse Caliper region timings from log files and emit "
                             "operator-breakdown CSVs (default: off)")
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Root directory for input/output (default: cwd)")
    args = parser.parse_args()

    sf = args.sf
    system = args.system

    base_dir = Path(args.base_dir) / args.result_dir
    out_dir = Path(args.base_dir) / "parse_caliper" / args.result_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    recall_dir = Path(args.base_dir) / "parse_caliper" / "recall"
    recall_dir.mkdir(parents=True, exist_ok=True)

    # Collect rows grouped by base_index
    rows_by_base_index = defaultdict(list)
    # Operator breakdown rows (only populated when --parse_caliper)
    caliper_rows_by_base_index = defaultdict(list)

    for sys_folder in [f"cpu-{system}", f"gpu-{system}"]:
        folder = base_dir / sys_folder / f"sf_{sf}"
        if not folder.is_dir():
            print(f"[SKIP] Folder not found: {folder}")
            continue

        for fname in sorted(os.listdir(folder)):
            # Try varbatch format first, then standard format as fallback
            m = FILENAME_RE.match(fname)
            is_standard = False
            if m:
                params_tag = None
            else:
                m = FILENAME_RE_STANDARD.match(fname)
                is_standard = True
                if not m:
                    continue

            query         = m.group("query")
            raw_query     = query  # before selectivity rename (caliper uses this)
            index         = m.group("index")
            device        = m.group("device")
            storage       = m.group("storage")
            index_storage = m.group("index_storage")
            file_sf       = m.group("sf")

            # Parse variant/params tags for index and query renaming
            params_tag = None
            if not is_standard:
                variant = m.group("variant") if "variant" in m.groupdict() and m.group("variant") else None
                parsed_tags = parse_variant_tags(variant)

                # Query rename (selectivity): new key-value format first, old format fallback
                sel_val = parsed_tags.get("sel", "")
                sel_tag = QUERY_VARIANT_MAP.get("sel", {}).get(sel_val, "")
                if not sel_tag:
                    sel_tag = extract_selectivity(variant)  # backward compat
                if sel_tag:
                    query = f"{query}_{sel_tag}"
            else:
                params_tag = m.group("params_tag") if "params_tag" in m.groupdict() and m.group("params_tag") else None
                parsed_tags = parse_variant_tags(params_tag) if params_tag else {}

            if file_sf != sf:
                continue

            case_key = (device, storage, index_storage)
            case_label = CASE_MAPPING.get(case_key)
            if case_label is None:
                print(f"[WARN] Unknown case combo {case_key} in {fname}")
                continue

            base_index = index.removeprefix("GPU,")
            # Index rename (cagra variant)
            base_index = apply_index_variant(base_index, parsed_tags)

            # Scenario flag (bs1_fullsweep needs per-qstart rows + setup-move time)
            scenario = m.group("scenario") if not is_standard and "scenario" in m.groupdict() else None
            is_bs1_fullsweep = (scenario == "bs1_fullsweep")

            filepath = folder / fname
            vsds_params  = parse_vsds_params(filepath)
            search_param = get_index_search_param(index, device, query, vsds_params)
            print(f"[PARSE] {filepath.name}")
            if is_standard:
                timings = parse_log_file_standard(filepath)
            elif is_bs1_fullsweep:
                timings = parse_log_file_bs1(filepath)
                setup_move_ms = parse_setup_index_movement_ms(filepath)
            else:
                timings = parse_log_file_caliper(filepath, raw_query)

            for t in timings:
                row = {
                    "query":        query,
                    "case":         case_label,
                    "batch_size":   t.get("batch_size"),   # None for standard-format rows
                    "params_tag":   params_tag,             # None for varbatch rows
                    "search_param": search_param,
                    "mean_ms":      t["mean_ms"],
                    "min_ms":     t["min_ms"],
                    "median_ms":  t["median_ms"],
                    "max_ms":     t["max_ms"],
                    "std_ms":     t["std_ms"],
                }
                if is_bs1_fullsweep:
                    row["qstart"] = t.get("qstart")
                    row["setup_index_movement_ms"] = setup_move_ms
                elif not is_standard:
                    row.update({
                        "mean_qps":   t["mean_qps"],
                        "min_qps":    t["min_qps"],
                        "median_qps": t["median_qps"],
                        "max_qps":    t["max_qps"],
                        "std_qps":    t["std_qps"],
                    })
                rows_by_base_index[base_index].append(row)

            # Caliper operator breakdown (varbatch files only)
            if args.parse_caliper and not is_standard:
                caliper_data = parse_caliper_regions_varbatch(filepath)
                if caliper_data:
                    op_rows = aggregate_operator_breakdown(
                        caliper_data, timings, query, case_label,
                        params_tag, search_param,
                        caliper_query=raw_query,
                    )
                    caliper_rows_by_base_index[base_index].extend(op_rows)

    if not rows_by_base_index:
        print("[ERROR] No data found. Check paths and --sf / --system arguments.")
        return

    # Determine which optional columns are present
    has_qps = any("mean_qps" in r for rows in rows_by_base_index.values() for r in rows)
    has_bs1_fullsweep = any("qstart" in r for rows in rows_by_base_index.values() for r in rows)
    timing_cols = ["query", "case", "batch_size", "params_tag", "search_param",
                   "mean_ms", "min_ms", "median_ms", "max_ms", "std_ms"]
    if has_qps:
        timing_cols += ["mean_qps", "min_qps", "median_qps", "max_qps", "std_qps"]
    if has_bs1_fullsweep:
        timing_cols += ["qstart", "setup_index_movement_ms"]

    # Sanitise result_dir for use in filenames (slashes → underscores)
    safe_result_dir = args.result_dir.replace("/", "_").replace("\\", "_")

    for base_index, rows in rows_by_base_index.items():
        df = pd.DataFrame(rows)
        df = df.reindex(columns=timing_cols)
        sort_cols = [c for c in ["query", "case", "batch_size", "qstart", "params_tag"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        # Sanitise base_index for filename (commas → underscores)
        safe_index = base_index.replace(",", "_").replace(" ", "_")
        out_csv = out_dir / f"{system}_{safe_result_dir}_{safe_index}_sf_{sf}.csv"
        df.to_csv(out_csv, index=False)
        print(f"[SAVED] {out_csv}  ({len(df)} rows)")

    # -----------------------------------------------------------------------
    # Operator breakdown CSVs (--parse_caliper only)
    # -----------------------------------------------------------------------
    if args.parse_caliper:
        breakdown_cols = (
            ["query", "case", "batch_size", "params_tag", "search_param", "stat"]
            + _BREAKDOWN_COLS
        )
        for base_index, op_rows in caliper_rows_by_base_index.items():
            if not op_rows:
                continue
            op_df = pd.DataFrame(op_rows)
            op_df = op_df.reindex(columns=breakdown_cols)
            sort_cols = [c for c in ["query", "case", "batch_size", "stat"] if c in op_df.columns]
            op_df = op_df.sort_values(sort_cols).reset_index(drop=True)
            safe_index = base_index.replace(",", "_").replace(" ", "_")
            op_csv = out_dir / f"operator_breakdown_{system}_{safe_result_dir}_{safe_index}_sf_{sf}.csv"
            op_df.to_csv(op_csv, index=False)
            print(f"[SAVED] {op_csv}  ({len(op_df)} rows)")
        if not caliper_rows_by_base_index:
            print("[CALIPER] No Caliper region data found in any log file.")

    # -----------------------------------------------------------------------
    # Recall computation — uses result CSVs at batch=10000
    # -----------------------------------------------------------------------
    recall_batch = SF_RECALL_BATCH.get(sf)
    if recall_batch is None:
        print(f"\n[RECALL] No recall batch size configured for sf={sf}. "
              f"Add an entry to SF_RECALL_BATCH to enable recall computation.")
    else:
        print(f"\n[RECALL] Computing recall@100 from batch={recall_batch} "
              f"(qstart=0) result CSVs for sf={sf}...")
        gt_dir_path = Path(f"./{args.gt_dir}")
        recall_rows = compute_all_recalls(base_dir, system, sf,
                                          recall_batch_size=recall_batch,
                                          include_all_gpu_cases=args.all_gpu_cases,
                                          gt_dir=gt_dir_path)
        if recall_rows:
            recall_df = pd.DataFrame(recall_rows,
                                     columns=["query", "case", "index_name", "params_tag",
                                              "mean_recall", "recall_k",
                                              "gt_filepath", "filepath"])
            recall_df = recall_df.sort_values(["query", "case", "index_name"]).reset_index(drop=True)
            recall_csv = recall_dir / f"{system}_{safe_result_dir}_recall_sf_{sf}.csv"
            recall_df.to_csv(recall_csv, index=False)
            print(f"[SAVED] {recall_csv}  ({len(recall_df)} rows)")
        else:
            print(f"[RECALL] No ANN CSV files found at batch={recall_batch}, qstart=0.")

    print("\nDone.")


if __name__ == "__main__":
    main()
