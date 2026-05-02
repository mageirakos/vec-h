import re
import os
from collections import defaultdict
import pandas as pd
import numpy as np
import math
from pathlib import Path
import argparse
import glob

# All recognized extra-tag keys (split by '-', key_value by '_')
ALL_EXTRA_KEYS = {"k", "metric", "sel", "cagra", "ivf", "vsd"}

# Keys that form the CSV filename suffix (data identity)
SUFFIX_KEYS = ["k", "metric"]  # ordered for deterministic filenames

# Index variant mappings: key -> {value -> index_rename_suffix}
INDEX_VARIANT_MAP = {
    "cagra": {"ch": "(C+H)", "c": "(C)"},
    "ivf":   {"h":  "(H)"},
}


# Query variant mappings: key -> {value -> query_suffix}
QUERY_VARIANT_MAP = {"sel": {"low": "low_sel", "high": "high_sel"}}

def parse_extra_tag(tag_str):
    """Parse extra tag string into a dict of recognized key-value pairs.

    The tag uses '-' to separate key-value groups and '_' within each group
    (key is the first token, value is the second token).
    Keys defined in ALL_EXTRA_KEYS are recognized; unrecognized keys are ignored.

    Examples:
      'k_100'                          -> {'k': '100'}
      'k_100-metric_IP'                -> {'k': '100', 'metric': 'IP'}
      'k_100-cagra_ch'                 -> {'k': '100', 'cagra': 'ch'}
      'k_100-metric_IP-cagra_ch'       -> {'k': '100', 'metric': 'IP', 'cagra': 'ch'}
      'k_100-batch_10000'              -> {'k': '100'}  ('batch' not recognized)
      'vary_batch_in_maxbench'         -> {}  ('vary' not recognized)
    """
    if not tag_str:
        return {}
    result = {}
    for group in tag_str.split("-"):
        parts = group.split("_")
        if len(parts) >= 2:
            key = parts[0].lower()
            if key in ALL_EXTRA_KEYS:
                result[key] = parts[1]
    return result

def rebuild_suffix_tag(parsed):
    """Rebuild extra tag string from suffix keys only (k, metric).
    Variant keys (cagra, sel) are stripped — they modify index/query names instead."""
    parts = []
    for key in SUFFIX_KEYS:
        if key in parsed:
            parts.append(f"{key}_{parsed[key]}")
    return "-".join(parts)

def normalize_bucket_tag(tag_str):
    """Return extra_tag with CASE-marker keys stripped — used as the CSV bucket key.

    Bucket-separating keys are exactly the ones that also affect the output CSV
    identity: SUFFIX_KEYS (filename) + INDEX_VARIANT_MAP / QUERY_VARIANT_MAP
    (display rename). Any other recognized key (currently just `vsd`) is a case
    marker by construction: it influences the case COLUMN but not the CSV itself.
    That way Case 1 ("k_100") and Case 4 ("k_100-vsd_cpu") share the same bucket
    and show up as two case columns of the same plot."""
    if not tag_str:
        return tag_str
    bucket_keys = set(SUFFIX_KEYS) | INDEX_VARIANT_MAP.keys() | QUERY_VARIANT_MAP.keys()
    kept = []
    for group in tag_str.split("-"):
        parts = group.split("_")
        if len(parts) >= 2 and parts[0].lower() in bucket_keys:
            kept.append(group)
    return "-".join(kept)

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
                # e.g. "GPU,IVF1024,Flat" -> "GPU,IVF1024(H),Flat"
                return re.sub(r'(IVF\d+)', rf'\1{suffix}', base_index)
    return base_index

parser=argparse.ArgumentParser(description="Parsing logs into excel tables.")

parser.add_argument("--sf", type=str, help="Scaling factor")
parser.add_argument("--benchmark", default="vsds", type=str, help="Benchmark(s) to run, comma-separated (e.g., 'vsds,tpch')")
parser.add_argument("--system", default="dgx-spark-02", type=str, help="System to run (default: dgx-spark-02)")
parser.add_argument("--incl_rep0", action="store_true", help="Include repetition 0 (do not skip warmup repetition)")
parser.add_argument("--base_dir", default=".", type=str,
                    help="Base directory for input logs and output CSVs (default: cwd)")

args=parser.parse_args()

reordered_schema = ["Total", "Operators", "Data Transfers", "Other", "IndexMovement", \
                    "Filter", "Project", "Join", "VectorSearch", "GroupBy", "OrderBy", "Distinct", "Limit", "LimitPerGroup", "Take", "LocalBroadcast", \
                    "Data Conversions", "Scatter", "Gather" ] # Optional: "TableSource", "TableSink", "Fused"

def get_expected_queries(benchmark):
    if benchmark == "tpch":
        return [f"q{i}" for i in range(1,23)]
    elif benchmark == "h2o":
        return [f"q{i}" for i in [1, 2, 3, 4, 5, 6, 7, 9, 10]]
    elif benchmark == "clickbench":
        return [f"q{i}" for i in [3, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 21, 22, 23, 24, 25, 26, 30, 31, 32, 33, 34, 35]]
    elif benchmark == "vsds":
        return ["q1_start", "q2_start", "q10_mid", "q11_end", "q13_mid", "q15_end", "q16_start", "q18_mid", "q19_start"]
    elif benchmark == "other-enn":
        return ["enn_reviews", "enn_reviews_project_distance", "ann_reviews", "enn_images", "enn_images_project_distance", "ann_images"]
    # TODO: handle the 'vary' cases and the 'batch' vs 'non-batch' cases. "The "EXTRA TAG"" should probably be passed in query name? To distinguish?
    elif benchmark in ["other-ann", "other-ann-batch", "other-ann-vary-ivf", "other-ann-vary-hnsw", "other-ann-vary-cagra"]:
        return ["ann_reviews", "ann_images"]
    elif benchmark == "other-enn-batch":
        return ["enn_reviews", "ann_reviews", "enn_images", "ann_images"]
    elif benchmark == "other-enn-ground-truth":
        return ["enn_reviews", "enn_images", "pre_reviews", "pre_images"]
    return []

def extract_repetitions(text):
    # Find all occurrences of "Repetition X" where X is a number
    matches = list(re.finditer(r'(^\s*Repetition_\d+)', text, re.MULTILINE))

    # If no repetitions are found, return an empty list
    if not matches:
        return []

    repetitions = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i + 1].start() if i + 1 < len(matches) else text.find("Generating")

        if end == -1:
            end = len(text)  # If "Generating" is not found, take the rest of the text

        repetitions.append(text[start:end].strip())

    return repetitions

def extract_executor_time(repetition):
    # The timings are usually in columns: [Inclusive] [Exclusive] ...
    # We want the second numeric value which represents the exclusive time for Executor::execute (or total execution time)
    match = re.search(r'Executor::execute\s+[\d\.]+\s+([\d\.]+)', repetition)
    return float(match.group(1)) if match else float('inf')

# Define a function to read the input file and process the data
def process_input_file(input_file_path, result, region_timings, query_name):
    with open(input_file_path, 'r') as file:
        content = file.read()

    # Split the content into sections based on the '[X/Y] Executing' pattern and capture table names
    sections_with_table_names = re.split(r'(\[\d+/\d+\] Executing.*?stats report)', content)

    reps = extract_repetitions(sections_with_table_names[0])
    # print("reps found =", len(reps))
    
    # By default we ignore the first repetition as warmup (if more than 1 exists).
    # If --incl_rep0 is passed, include repetition 0 as well.
    if args.incl_rep0:
        valid_reps = reps
    else:
        valid_reps = reps[1:] if len(reps) > 1 else reps
    if not valid_reps:
        return

    regions = ["Executor::execute", "OPERATORS", "_CUDF_", "_ACERO_", "_NATIVE_", "_FAISS_", "DataTransformation", "CPU->GPU", "GPU->CPU", "CPU->CPU", "GPU->GPU", "index_movement"]

    # We process each valid repetition and store its timings
    for rep_idx, rep_content in enumerate(valid_reps):
        # We simulate the structure of sections logic
        lines = rep_content.splitlines()
        in_execute_block = False
        
        # Temporary dict to hold timings for this specific repetition
        temp_timings = defaultdict(float)

        for line in lines:
            if "Executor::execute" in line:
                in_execute_block = True
            elif "Executor::schedule" in line:
                in_execute_block = False
            
            if not in_execute_block:
                continue

            for region in regions:
                if region in line:
                    clean_line = line.replace("’", "")
                    cols = clean_line.split()
                    region_name = cols[0]
                    temp_timings[region_name] += float(cols[2]) * 1000.0
        
        # Now append these temp timings to our global query timings list
        for r_name, r_val in temp_timings.items():
            if query_name not in region_timings:
                region_timings[query_name] = defaultdict(list)
            region_timings[query_name][r_name].append(r_val)


def get_value(value, in_ns = False):
    if value is None:
        return 0.0

    value = float(value)

    if math.isnan(value):
        return 0.0

    if in_ns:
        return value #  * 1e-6

    return value

# Helper to process scalar or array values
def compute_stats(values_list, in_ns=False):
    if not values_list:
        return {"mean": 0.0, "min": 0.0, "median": 0.0, "max": 0.0, "std": 0.0}
    
    # Optional scaling
    vals = np.array(values_list, dtype=float)
    if in_ns:
        pass # Handle if needed (was commented out in original)
        
    return {
        "mean": np.mean(vals),
        "min": np.min(vals),
        "median": np.median(vals),
        "max": np.max(vals),
        "std": np.std(vals) if len(vals) > 1 else 0.0
    }

def classify_region(region_key, val):
    """Classify a single region_key/value into timing categories. Returns dict of increments."""
    increments = defaultdict(float)
    operators_time = 0.0
    # NOTE: basically Caliper "operators" includes data movement, here we breakt it into actual operators and data movement separately
    # index transfer is part of the total data movement, so the diff is table movement.
    if "Executor::execute" in region_key:
        increments['Total'] += val
        increments['_overhead'] += val  # track overhead separately

    if "CPU->GPU" in region_key or "GPU->CPU" in region_key:
        increments["Data Transfers"] += val

    if "OPERATORS" in region_key:
        increments['Operators'] += val
        increments['_overhead'] -= val  # subtract from overhead

    if "CPU->CPU" in region_key or "GPU->GPU" in region_key:
        increments['Data Conversions'] += val

    if "index_movement" in region_key:
        increments["IndexMovement"] += val

    if "DISTINCT" in region_key:
        increments['Distinct'] += val
        increments['_ops_time'] += val

    if "SCATTER" in region_key:
        increments['Scatter'] += val
        increments['_ops_time'] += val

    if "GATHER" in region_key:
        increments['Gather'] += val
        increments['_ops_time'] += val

    if "FILTER" in region_key:
        increments['Filter'] += val
        increments['_ops_time'] += val

    if "PROJECT" in region_key:
        increments['Project'] += val
        increments['_ops_time'] += val

    if "JOIN" in region_key:
        increments['Join'] += val
        increments['_ops_time'] += val

    if "VECTOR_SEARCH" in region_key:
        increments['VectorSearch'] += val
        increments['_ops_time'] += val

    if "GROUP_BY" in region_key:
        increments['GroupBy'] += val
        increments['_ops_time'] += val

    if "ORDER_BY" in region_key:
        increments['OrderBy'] += val
        increments['_ops_time'] += val

    # We should be including the time from this in the "limit"
    # if "LIMIT_PER_GROUP" in region_key :
    #     increments['Limit'] += val
    #     increments['_ops_time'] += val

    if "LIMIT" in region_key :
        increments['Limit'] += val
        increments['_ops_time'] += val

    if "TAKE" in region_key:
        increments['Take'] += val
        increments['_ops_time'] += val

    if "LOCAL_BROADCAST" in region_key:
        increments['LocalBroadcast'] += val
        increments['_ops_time'] += val

    return increments

# Function to format the results as required
def _compute_per_rep_timings(values):
    """Build a list of per-repetition query_timings dicts.

    Each dict has all the named region buckets plus the derived columns
    (Total, Other) computed *for that rep* — i.e. before any aggregation.
    This is the right place to compute differences/sums over correlated
    regions, so that downstream stats are stat(derived) instead of
    derived(stats).
    """
    if not values:
        return []
    n_reps = max(len(v) for v in values.values())

    per_rep = []
    for rep_idx in range(n_reps):
        qt = defaultdict(float)
        for region in reordered_schema:
            qt[region] = 0
        qt["VectorSearch"] = 0
        qt["IndexMovement"] = 0

        for region_key, vals_list in values.items():
            val = vals_list[rep_idx] if rep_idx < len(vals_list) else 0.0
            for k, v in classify_region(region_key, val).items():
                qt[k] += v

        # 'Total' was populated from Executor::execute(I) by classify_region.
        # Use it as the per-rep ground truth so Total ≡ Execute by construction.
        execute_time = qt['Total']
        qt.pop('_overhead', None)  # no longer used
        operators_time = qt.pop('_ops_time', 0.0)
        qt['Operators'] = operators_time
        # 'Other' is the residual: everything inside Executor::execute that
        # isn't a named leaf operator bucket and isn't a data transfer.
        # It includes:
        #   - Execute-region exclusive (executor pre/post-op wrapper, ~1-2 ms)
        #   - OPERATORS-region exclusive (operator-tree wrapper)
        #   - CPU/GPU group exclusive (push-based pipeline plumbing between
        #     operators — `add_input`/`export_next_batch` orchestration that
        #     runs outside any leaf operator's instrumented region)
        #   - DataTransformation parent exclusive (bookkeeping above CPU->CPU
        #     / CPU->GPU children)
        #   - CPU_NATIVE_TABLE_SOURCE / CPU_NATIVE_TABLE_SINK (no classify rule)
        #   - CPU_GPU_SYNC (not in parser's regions list)
        #   - Data Conversions (CPU->CPU + GPU->GPU format conversions —
        #     these are also reported separately in the 'Data Conversions'
        #     column for inspection)
        # Defined as residual so Total ≡ Operators + Data Transfers + Other.
        qt['Other'] = execute_time - operators_time - qt['Data Transfers']
        per_rep.append(qt)

    return per_rep

def print_results(result, region_timings, queries, storage_devices):
    final_table = defaultdict(lambda: defaultdict(float))

    for query_name in queries:
        values = region_timings[query_name]
        per_rep_timings = _compute_per_rep_timings(values)

        # Collect every column that appeared in any rep
        all_cols = set()
        for qt in per_rep_timings:
            all_cols.update(qt.keys())

        for stat in ["mean", "min", "median", "max", "std"]:
            query_timings = defaultdict(float)
            for region in reordered_schema:
                query_timings[region] = 0
            query_timings["VectorSearch"] = 0
            query_timings["IndexMovement"] = 0

            # Aggregate each column across reps — derived columns (Total, Other)
            # are already per-rep, so this is stat(per-rep value), not a
            # combination of stats from different regions.
            for col in all_cols:
                col_vals = [qt.get(col, 0.0) for qt in per_rep_timings]
                col_stats = compute_stats(col_vals, True)
                query_timings[col] = col_stats[stat]

            # Combine the metric type into the query name for the row index
            row_key = f"{query_name}_{stat}"
            final_table[row_key] = query_timings

            # Sanity check: Other should never be negative. A negative Other
            # would mean some operator bucket is double-counted in classify_region
            # (e.g., a region key matching two operator substrings).
            if stat in ["mean", "min", "median"]:
                if query_timings.get('Other', 0.0) < -1e-3:
                    print(f"[WARNING] Query {row_key} Other is negative ({query_timings['Other']:.3f} ms) — possible double-counting in classify_region")

    df = pd.DataFrame.from_dict(final_table, orient="index")
    df.index.name = "Query_Metric"
    df.columns.name = "Region"

    # Sort the index such that all _min are together, then _mean, _median, _max, _std
    # and within each metric, sort queries logically by number (q1, q2, q10, etc.)
    metric_order = {"min": 0, "mean": 1, "median": 2, "max": 3, "std": 4}
    
    import re
    def get_sort_key(idx_val):
        parts = str(idx_val).rsplit('_', 1)
        if len(parts) == 2 and parts[1] in metric_order:
            query, metric = parts
            m = re.match(r'^q(\d+)', query)
            q_num = int(m.group(1)) if m else 999
            return (metric_order[metric], q_num, query)
        return (99, 999, idx_val)
        
    df = df.reindex(sorted(df.index, key=get_sort_key))

    print(df)
    return df

def print_per_rep_results(region_timings, queries):
    """Produce per-repetition rows: <query>_rep1, <query>_rep2, etc."""
    final_table = defaultdict(lambda: defaultdict(float))

    for query_name in queries:
        values = region_timings[query_name]
        per_rep_timings = _compute_per_rep_timings(values)
        if not per_rep_timings:
            continue

        for rep_idx, query_timings in enumerate(per_rep_timings):
            # Use zero-based rep numbering when rep0 is included, otherwise keep legacy 1-based naming
            if globals().get('args', None) and getattr(args, 'incl_rep0', False):
                row_key = f"{query_name}_rep{rep_idx}"
            else:
                row_key = f"{query_name}_rep{rep_idx + 1}"
            final_table[row_key] = query_timings

    df = pd.DataFrame.from_dict(final_table, orient="index")
    df.index.name = "Query_Rep"
    df.columns.name = "Region"

    # Sort by query number then rep number
    def get_sort_key(idx_val):
        m = re.match(r'^(.*?)_rep(\d+)$', str(idx_val))
        if m:
            query = m.group(1)
            rep_num = int(m.group(2))
            qm = re.match(r'^q(\d+)', query)
            q_num = int(qm.group(1)) if qm else 999
            return (q_num, query, rep_num)
        return (999, str(idx_val), 0)

    df = df.reindex(sorted(df.index, key=get_sort_key))
    return df

def parse_device(folder_path, device, storage_device, index_storage_device, vs_device, index_desc, sf, benchmark):
    """Scan folder_path and return {bucket_tag: (df, df_per_rep)} for all extra_tag variants found.

    `vs_device` filters files by their `vsd` extra-tag value (defaults to `device` when
    the file has no `vsd_*` tag). Bucket key strips CASE_MARKER_KEYS from the extra_tag
    so same-CSV aggregation keeps Case 4 alongside other cases."""
    expected_queries = get_expected_queries(benchmark)
    sf_escaped = sf.replace(".", "\\.")
    pattern = re.compile(
        rf"q_(?P<query>.*?)-i_(?P<index>.*?)-d_{device}-s_{storage_device}-is_{index_storage_device}-sf_{sf_escaped}(?:-(?P<extra>.*?))?\.log"
    )

    # Group files by bucket_tag (extra_tag minus CASE_MARKER_KEYS) -> {query: filepath}
    files_by_extra = defaultdict(dict)
    if os.path.isdir(folder_path):
        for f in os.listdir(folder_path):
            match = pattern.match(f)
            if match and match.group("index") == index_desc:
                extra_tag = match.group("extra") or ""
                parsed_tag = parse_extra_tag(extra_tag)
                # Case is identified by `vsd` tag; absent => same as outer device.
                file_vs_device = parsed_tag.get("vsd", device)
                if file_vs_device != vs_device:
                    continue
                bucket_tag = normalize_bucket_tag(extra_tag)
                files_by_extra[bucket_tag][match.group("query")] = os.path.join(folder_path, f)

    if not files_by_extra:
        print(f"[DEBUG] No files found in {folder_path} for device={device}, storage={storage_device}, index_storage={index_storage_device}, vs_device={vs_device}, index={index_desc}")
        return {}

    results = {}
    for extra_tag, found_in_folder in sorted(files_by_extra.items()):
        print(f"[DEBUG] Found files in {folder_path} for extra_tag='{extra_tag}': {list(found_in_folder.keys())}")
        result = {}
        region_timings = defaultdict(lambda: defaultdict(list))

        # Use discovered queries from files; fall back to hardcoded list if all expected are present
        discovered_queries = sorted(found_in_folder.keys())
        if set(expected_queries).issubset(set(discovered_queries)):
            # All expected queries found — use the expected list (preserves canonical order)
            queries_to_process = expected_queries
        else:
            # Not all expected queries found — use what we discovered from filenames
            queries_to_process = discovered_queries

        for query in queries_to_process:
            if query in found_in_folder:
                process_input_file(found_in_folder[query], result, region_timings, query)
            else:
                print(f"WARNING: No file found for query={query}, extra_tag='{extra_tag}' in {folder_path}")

        df = print_results(result, region_timings, queries_to_process, [storage_device])
        df_per_rep = print_per_rep_results(region_timings, queries_to_process)
        results[extra_tag] = (df, df_per_rep)

    return results

# Main function
if __name__ == "__main__":
    # Path to the folder containing the input files
    system = args.system
    sf = args.sf
    benchmark_list = [b.strip() for b in args.benchmark.split(',')]
    all_benchmark_tables = []
    
    for benchmark in benchmark_list:
        print(f"\n{'='*20}")
        print(f"STARTING BENCHMARK: {benchmark}")
        print(f"{'='*20}\n")
        
        base_folder = os.path.join(args.base_dir, benchmark)
        tables = [] # Local tables for this specific benchmark

        if benchmark == "vsds" or benchmark.startswith("other-"):
            # Discover base index types using the new tagged format
            # q_<query>-i_<index>-d_<device>-s_<storage>-is_<index_storage>-sf_<sf>.log
            all_base_indexes = set()
            file_pattern = re.compile(r"q_.*?-i_(?P<index>.*?)-d_.*?-s_.*?-is_.*?-sf_.*\.log")

            for sys_folder in [f"cpu-{system}", f"gpu-{system}"]:
                folder_path = os.path.join(base_folder, sys_folder, f"sf_{sf}")
                if os.path.isdir(folder_path):
                    for f in os.listdir(folder_path):
                        match = file_pattern.match(f)
                        if match:
                            idx = match.group("index")
                            # Stick to existing grouping: GPU,Flat and Flat treated as same base index
                            idx_base = idx.replace("GPU,", "")
                            all_base_indexes.add(idx_base)

            print(f"Discovered index types: {all_base_indexes}")

            for base_index in all_base_indexes:
                print(f"\n--- Processing index type: {base_index} ---")

                combos = [
                    # device, storage_device, index_storage_device, vs_device, index_desc, sys_folder, label
                    ("cpu", "cpu", "cpu", "cpu", base_index, f"cpu-{system}", "0: CPU-CPU-CPU"),
                    # Fallback for GPU-only indexes (e.g. Cagra) whose CPU-case logs still use the GPU, prefix
                    ("cpu", "cpu", "cpu", "cpu", f"GPU,{base_index}", f"cpu-{system}", "0: CPU-CPU-CPU"),
                    ("gpu", "cpu", "cpu", "gpu", f"GPU,{base_index}", f"gpu-{system}", "1: GPU-CPU-CPU"),
                    ("gpu", "cpu-pinned", "cpu-pinned", "gpu", f"GPU,{base_index}", f"gpu-{system}", "1P: GPU-CPU(P)-CPU(P)"),
                    ("gpu", "cpu", "gpu", "gpu", f"GPU,{base_index}", f"gpu-{system}", "2: GPU-CPU-GPU"),
                    ("gpu", "gpu", "gpu", "gpu", f"GPU,{base_index}", f"gpu-{system}", "3: GPU-GPU-GPU"),
                    # Case 4: GPU relational + CPU vector-search (hybrid).
                    # Identified explicitly via the `vsd_cpu` extra-tag marker on the
                    # filename (parse_device filters by vs_device from the parsed tag).
                    ("gpu", "cpu", "cpu", "cpu", base_index, f"gpu-{system}", "4: MIXED(VS=CPU)"),
                    # Fallback for GPU-only descriptors (e.g. "GPU,Cagra") that the engine
                    # converts to CPU (HNSWCagra) at setup because index_storage_device=cpu.
                    ("gpu", "cpu", "cpu", "cpu", f"GPU,{base_index}", f"gpu-{system}", "4: MIXED(VS=CPU)"),
                ]

                # Collect results per bucket_tag across all combos
                all_dfs = defaultdict(dict)       # bucket_tag -> {label: df}
                all_dfs_per_rep = defaultdict(dict)  # bucket_tag -> {label: df_per_rep}

                for device, storage, index_storage, vs_device, index_desc, sys_folder, label in combos:
                    folder_path = os.path.join(base_folder, sys_folder, f"sf_{sf}")
                    if os.path.isdir(folder_path):
                        print(f"Processing: {folder_path} for Device={device}, Storage={storage}, IndexStorage={index_storage}, VSDev={vs_device}, Index={index_desc}")
                        results_by_extra = parse_device(folder_path, device, storage, index_storage, vs_device, index_desc, sf, benchmark)
                        for extra_tag, (df, df_per_rep) in results_by_extra.items():
                            print(f"DataFrame for {label} (extra_tag='{extra_tag}') has shape {df.shape}")
                            if not df.empty:
                                all_dfs[extra_tag][label] = df
                            if not df_per_rep.empty:
                                all_dfs_per_rep[extra_tag][label] = df_per_rep
                    else:
                        print(f"Folder not found: {folder_path} for combination")

                if not all_dfs:
                    print(f"No data found for index type {base_index} in benchmark {benchmark}. Skipping.")
                    continue

                os.makedirs(os.path.join(args.base_dir, "parse_caliper", benchmark), exist_ok=True)
                per_rep_dir = os.path.join(args.base_dir, "parse_caliper", "per_rep", benchmark)
                os.makedirs(per_rep_dir, exist_ok=True)

                # Fill in missing CPU case for variant indexes (CagraCH, IVFH).
                # On CPU (case 0), the H/CH optimizations have no effect — results
                # are identical to the base variant. Copy the base data so plots
                # show all 4 cases without running pointless duplicate benchmarks.
                cpu_label = "0: CPU-CPU-CPU"
                # Find the base tag: the one with no variant keys (only suffix keys like k, metric).
                variant_keys = set(INDEX_VARIANT_MAP.keys()) | set(QUERY_VARIANT_MAP.keys())
                base_tag = None
                for tag in all_dfs:
                    parsed_tag = parse_extra_tag(tag)
                    if not (set(parsed_tag.keys()) & variant_keys):
                        if cpu_label in all_dfs[tag]:
                            base_tag = tag
                            break
                if base_tag is not None and cpu_label in all_dfs[base_tag]:
                    for vtag in list(all_dfs.keys()):
                        if vtag and vtag != base_tag and cpu_label not in all_dfs[vtag]:
                            print(f"[INFO] Copying CPU case from base to variant '{vtag}' (optimization has no CPU effect)")
                            all_dfs[vtag][cpu_label] = all_dfs[base_tag][cpu_label]
                            if base_tag in all_dfs_per_rep and cpu_label in all_dfs_per_rep[base_tag]:
                                if vtag not in all_dfs_per_rep:
                                    all_dfs_per_rep[vtag] = {}
                                all_dfs_per_rep[vtag][cpu_label] = all_dfs_per_rep[base_tag][cpu_label]

                for extra_tag in sorted(all_dfs.keys()):
                    # Parse extra tag: extract variant keys (cagra, sel) and suffix keys (k, metric)
                    parsed = parse_extra_tag(extra_tag)
                    display_index = apply_index_variant(base_index, parsed)
                    suffix_str = rebuild_suffix_tag(parsed)
                    tag_suffix = f"_{suffix_str}" if suffix_str else ""

                    dataframes = all_dfs[extra_tag]
                    dataframes_per_rep = all_dfs_per_rep.get(extra_tag, {})

                    if dataframes:
                        df_merged = pd.concat(dataframes, axis=1)
                        df_merged = df_merged.swaplevel(axis=1).sort_index(axis=1) if isinstance(df_merged.columns, pd.MultiIndex) else df_merged
                        cols_to_keep = [col for col in df_merged.columns if col[0] in reordered_schema]
                        df_merged = df_merged[cols_to_keep]

                        ordered_cols = []
                        for r in reordered_schema:
                            ordered_cols.extend([c for c in cols_to_keep if c[0] == r])
                        df_merged = df_merged.reindex(columns=ordered_cols)
                        df_merged = df_merged.round(6)

                        output_csv = f"{args.system}_{benchmark}_{display_index}_sf_{sf}{tag_suffix}.csv"
                        df_merged.to_csv(os.path.join(args.base_dir, "parse_caliper", benchmark, output_csv))
                        tables.append((f"{args.system}_{benchmark}_{display_index}_sf_{sf}{tag_suffix}", df_merged))

                        print("=====================================")
                        print(f"Combined Results for {display_index} (extra_tag='{extra_tag}') saved to {output_csv}")
                        print("=====================================")

                    if dataframes_per_rep:
                        df_per_rep_merged = pd.concat(dataframes_per_rep, axis=1)
                        df_per_rep_merged = df_per_rep_merged.swaplevel(axis=1).sort_index(axis=1) if isinstance(df_per_rep_merged.columns, pd.MultiIndex) else df_per_rep_merged
                        cols_to_keep_pr = [col for col in df_per_rep_merged.columns if col[0] in reordered_schema]
                        df_per_rep_merged = df_per_rep_merged[cols_to_keep_pr]

                        ordered_cols_pr = []
                        for r in reordered_schema:
                            ordered_cols_pr.extend([c for c in cols_to_keep_pr if c[0] == r])
                        df_per_rep_merged = df_per_rep_merged.reindex(columns=ordered_cols_pr)
                        df_per_rep_merged = df_per_rep_merged.round(6)

                        per_rep_csv = f"{args.system}_{benchmark}_{display_index}_sf_{sf}{tag_suffix}.csv"
                        df_per_rep_merged.to_csv(os.path.join(per_rep_dir, per_rep_csv))
                        print(f"Per-rep results saved to {per_rep_dir}{per_rep_csv}")

    output_file = os.path.join(args.base_dir, f"{args.system}-csv-{sf}.xlsx")
    if tables:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            start_row = 0
            for name, df in tables:
                df.to_excel(writer, sheet_name=f"{args.system}-csv-{sf}", startrow=start_row, startcol=0)
                start_row += len(df) + 10


