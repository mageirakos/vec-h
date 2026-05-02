#!/usr/bin/env python3
"""
Recall calculator for VSDS and vary-batch benchmark CSV outputs.

Compares query result CSVs against ground-truth baselines and computes:
  - Relaxed Recall: fraction of GT rows present in the target (order-independent)
  - Strict Recall:  fraction of GT rows matching at the same position (order-dependent)

Ground Truth (GT) selection:
  - CPU GT: i_Flat, d_cpu-s_cpu-is_cpu   (case 0)
  - GPU GT: i_GPU,Flat, d_gpu-s_cpu-is_cpu (case 1, the simplest GPU config)
  - Fallback GT: if no GT in this query group, try mapped query (post_→pre_, ann_→enn_)

Groups by (query, sf, k, sel) so selectivity variants get separate GT.
Computes recall for ALL execution cases (not just case 0 and 3).

Usage:
  python recall_vsds.py --results_dir ./vsds --recall_dir ./recall/vsds --sf 1 --system dgx-spark-02

Manual run against an existing run (Maximus intra-engine):
  cd /local/home/vmageirakos/projects/Maximus
  python results/recall_vsds.py \
    --results_dir results/vsds \
    --recall_dir  results/recall \
    --sf 1 --system dgx-spark-02

Output: A CSV with columns:
  query, sf, sel, case, device, index, relaxed_recall, strict_recall, gt_rows, target_rows, gt_source, gt_filepath, filepath
"""

import argparse
import os
import re
import sys
from collections import defaultdict

# =============================================================================
# 1. Filename parsing
# =============================================================================

# Pattern: maximus_q_<query>-i_<index>-d_<device>-s_<storage>-is_<index_storage>-sf_<sf>[-<extra>].csv
FILE_PATTERN = re.compile(
    r"maximus_q_(?P<query>.*?)"
    r"-i_(?P<index>.*?)"
    r"-d_(?P<device>.*?)"
    r"-s_(?P<storage>.*?)"
    r"-is_(?P<index_storage>.*?)"
    r"-sf_(?P<sf>.*?)"
    r"(?:-(?P<extra>.*?))?"
    r"\.csv$"
)


def parse_csv_filename(filename):
    """Parse a benchmark CSV filename into its component fields.
    Returns a dict or None if the filename doesn't match the expected pattern.
    """
    m = FILE_PATTERN.match(filename)
    if not m:
        return None
    info = m.groupdict()
    info["extra"] = info.get("extra") or ""
    # Parse known tag keys from extra (e.g. 'k_100-metric_IP-sel_low')
    # Maxbench appends '_batch_N_qstart_M' with underscores, which bleeds into
    # the last '-'-delimited group. Strip it before extracting tag values.
    info["k"] = ""
    info["sel"] = ""
    info["batch"] = ""
    if info["extra"]:
        # Extract batch from anywhere in the extra string (always _batch_N pattern)
        batch_m = re.search(r'_batch_(\d+)', info["extra"])
        if batch_m:
            info["batch"] = batch_m.group(1)
        for group in info["extra"].split("-"):
            # Strip _batch_*_qstart_* suffix that maxbench appends
            clean = re.sub(r'_batch_\d+(?:_qstart_\d+)?$', '', group)
            parts = clean.split("_", 1)
            if len(parts) == 2:
                if parts[0] == "k":
                    info["k"] = parts[1]
                elif parts[0] == "sel":
                    info["sel"] = parts[1]
    return info


# =============================================================================
# 2. Row normalization (float truncation for hardware noise)
# =============================================================================

FLOAT_TRUNC_RE = re.compile(r"(\.\d{4})\d+")


def normalize_row(row_str):
    """Truncate floats to 4 decimal places to absorb hardware noise."""
    return FLOAT_TRUNC_RE.sub(r"\1", row_str).strip()


def load_rows(filepath):
    """Load CSV skipping header, return list of normalized row strings."""
    with open(filepath, "r") as f:
        lines = f.readlines()
    if len(lines) <= 1:
        return []
    # Skip header (line 0), normalize remaining
    return [normalize_row(line) for line in lines[1:] if line.strip()]


# =============================================================================
# 3. Recall computation
# =============================================================================


# Q19 is a scalar SUM(revenue); strict/relaxed recall still applies on the
# 1-row output, but we ALSO compute a revenue-error so you can see how far
# an ANN index is from the exact answer in absolute / relative terms.
def q19_revenue_error(gt_rows, target_rows):
    """Return (abs_err, rel_err) for Q19's single-row revenue output, or
    (None, None) if either side isn't a valid 1-row revenue CSV."""
    try:
        rev_exact = float(gt_rows[0])
        rev_got   = float(target_rows[0])
    except (IndexError, ValueError):
        return (None, None)
    abs_err = rev_got - rev_exact
    rel_err = abs(abs_err) / rev_exact if rev_exact else float("nan")
    return (abs_err, rel_err)


def compute_recall(gt_rows, target_rows):
    """Compute relaxed and strict recall.

    Relaxed recall:  |GT ∩ Target| / |GT|  (set-based, order-independent)
    Strict recall:   count of positions where GT[i] == Target[i], / |GT|

    Returns (relaxed_recall, strict_recall, match_relaxed, match_strict)
    """
    n_gt = len(gt_rows)
    if n_gt == 0:
        return (1.0, 1.0, 0, 0) if len(target_rows) == 0 else (0.0, 0.0, 0, 0)

    # --- Relaxed: multiset intersection ---
    # Use a multiset (dict of counts) so duplicate rows are handled correctly
    from collections import Counter
    gt_counter = Counter(gt_rows)
    target_counter = Counter(target_rows)
    match_relaxed = 0
    for row, count in gt_counter.items():
        match_relaxed += min(count, target_counter.get(row, 0))

    # --- Strict: positional match ---
    match_strict = 0
    for i in range(min(n_gt, len(target_rows))):
        if gt_rows[i] == target_rows[i]:
            match_strict += 1

    relaxed_recall = match_relaxed / n_gt
    strict_recall = match_strict / n_gt
    return (relaxed_recall, strict_recall, match_relaxed, match_strict)


# =============================================================================
# 4. Discovery & ground-truth selection
# =============================================================================


def discover_csv_files(results_dir):
    """Walk results_dir and return list of (filepath, parsed_info) tuples."""
    files = []
    for dirpath, _, filenames in os.walk(results_dir):
        for fname in filenames:
            if not fname.endswith(".csv"):
                continue
            info = parse_csv_filename(fname)
            if info is None:
                continue
            info["filepath"] = os.path.join(dirpath, fname)
            info["filename"] = fname
            files.append(info)
    return files


def is_cpu_gt(info):
    """CPU ground truth: Flat index on cpu-cpu-cpu."""
    return (
        info["index"] == "Flat"
        and info["device"] == "cpu"
        and info["storage"] == "cpu"
        and info["index_storage"] == "cpu"
    )


def is_gpu_gt(info):
    """GPU ground truth: GPU,Flat index, d_gpu-s_cpu-is_cpu (case 1)."""
    return (
        info["index"] == "GPU,Flat"
        and info["device"] == "gpu"
        and info["storage"] == "cpu"
        and info["index_storage"] == "cpu"
    )


def group_key(info):
    """Group files by (query, sf, k, sel) so we compare like-for-like.
    Selectivity must be in the key so low/high variants get separate GT."""
    return (info["query"], info["sf"], info["k"], info["sel"])


def is_case0(info):
    """Case 0: all-CPU (d_cpu-s_cpu-is_cpu)."""
    return info["device"] == "cpu" and info["storage"] == "cpu" and info["index_storage"] == "cpu"


def is_case3(info):
    """Case 3: all-GPU (d_gpu-s_gpu-is_gpu)."""
    return info["device"] == "gpu" and info["storage"] == "gpu" and info["index_storage"] == "gpu"


def query_sort_key(query_str):
    """Sort queries by numeric part then suffix (e.g. q2_start < q10_mid < q11_end)."""
    m = re.match(r"q(\d+)", query_str)
    num = int(m.group(1)) if m else 999
    return (num, query_str)


# Fallback GT query mapping: post→pre (same logical result with Flat index).
# Used when no GT is found within the same query group (e.g. post_images has no
# Flat run, but pre_images does — both return exact NN on the filtered subset).
_GT_QUERY_FALLBACK_PREFIXES = [
    ("post_", "pre_"),
    ("ann_",  "enn_"),
]


def _get_fallback_query(query):
    """Map a query name to its fallback GT query (e.g. post_images → pre_images)."""
    for prefix, fallback_prefix in _GT_QUERY_FALLBACK_PREFIXES:
        if query.startswith(prefix):
            return query.replace(prefix, fallback_prefix, 1)
    return None


def _case_label(info):
    """Human-readable case label from device/storage/index_storage."""
    d, s, isd = info["device"], info["storage"], info["index_storage"]
    return f"d_{d}-s_{s}-is_{isd}"


# =============================================================================
# 5. Main logic
# =============================================================================


def run(args):
    all_files = discover_csv_files(args.results_dir)
    if not all_files:
        print(f"No CSV files found in {args.results_dir}")
        sys.exit(1)

    # Apply filters
    if args.sf:
        all_files = [f for f in all_files if f["sf"] == args.sf]
    if args.query:
        all_files = [f for f in all_files if f["query"] == args.query]
    if args.k:
        all_files = [f for f in all_files if f["k"] == args.k]

    if args.system:
        all_files = [f for f in all_files if args.system in f["filepath"]]
    if args.batch_filter:
        all_files = [f for f in all_files if f["batch"] == args.batch_filter]

    # Group by (query, sf, k, sel)
    groups = defaultdict(list)
    for info in all_files:
        groups[group_key(info)].append(info)

    results = []
    warnings = []

    # Sort groups by query number
    sorted_keys = sorted(groups.keys(), key=lambda g: query_sort_key(g[0]))

    for (query, sf, k, sel) in sorted_keys:
        file_list = groups[(query, sf, k, sel)]
        sel_tag = f", sel={sel}" if sel else ""

        # Find GTs within this group
        cpu_gt_list = [f for f in file_list if is_cpu_gt(f)]
        gpu_gt_list = [f for f in file_list if is_gpu_gt(f)]

        cpu_gt = cpu_gt_list[0] if cpu_gt_list else None
        gpu_gt = gpu_gt_list[0] if gpu_gt_list else None

        # Fallback: try mapped query (e.g. post_images → pre_images)
        gt_source = "direct"
        if not cpu_gt and not gpu_gt:
            fallback_query = _get_fallback_query(query)
            if fallback_query:
                fallback_key = (fallback_query, sf, k, sel)
                fallback_files = groups.get(fallback_key, [])
                cpu_gt_list = [f for f in fallback_files if is_cpu_gt(f)]
                gpu_gt_list = [f for f in fallback_files if is_gpu_gt(f)]
                cpu_gt = cpu_gt_list[0] if cpu_gt_list else None
                gpu_gt = gpu_gt_list[0] if gpu_gt_list else None
                if cpu_gt or gpu_gt:
                    gt_source = f"fallback:{fallback_query}"
                    print(f"  [fallback-gt] Using {fallback_query} as GT for {query} (sf={sf}, k={k}{sel_tag})")

        if not cpu_gt and not gpu_gt:
            warnings.append(f"[WARN] No ground truth found for query={query}, sf={sf}, k={k}{sel_tag}")
            continue

        # Cross-check: CPU GT vs GPU GT should be identical for Flat
        if cpu_gt and gpu_gt:
            cpu_rows = load_rows(cpu_gt["filepath"])
            gpu_rows = load_rows(gpu_gt["filepath"])
            rel, strict, _, _ = compute_recall(cpu_rows, gpu_rows)
            if rel < 1.0 or strict < 1.0:
                warnings.append(
                    f"[WARN] CPU GT vs GPU GT MISMATCH for query={query}, sf={sf}, k={k}{sel_tag}: "
                    f"relaxed={rel:.4f}, strict={strict:.4f}"
                )

        # Compare every non-GT file against the appropriate GT.
        # CPU-device targets → CPU GT; GPU-device targets → GPU GT (fallback to CPU GT).
        for target in file_list:
            if is_cpu_gt(target) or is_gpu_gt(target):
                continue  # GT files are not targets

            # Pick the right GT based on device
            if target["device"] == "cpu":
                gt = cpu_gt
            else:
                gt = gpu_gt if gpu_gt else cpu_gt

            if gt is None:
                warnings.append(
                    f"[WARN] No matching GT for target {target['filename']} "
                    f"(query={query}, sf={sf}, k={k}{sel_tag})"
                )
                continue

            gt_rows = load_rows(gt["filepath"])
            target_rows = load_rows(target["filepath"])
            rel, strict, match_rel, match_strict = compute_recall(gt_rows, target_rows)

            # Q19-only: extra revenue-error metric (set-recall is degenerate on 1 row).
            rev_abs, rev_rel = (q19_revenue_error(gt_rows, target_rows)
                                if query == "q19_start" else (None, None))

            results.append({
                "query": query,
                "sf": sf,
                "sel": sel,
                "case": _case_label(target),
                "device": target["device"],
                "index": target["index"],
                "relaxed_recall": round(rel, 6),
                "strict_recall": round(strict, 6),
                "rev_err_abs": "" if rev_abs is None else round(rev_abs, 4),
                "rev_err_rel": "" if rev_rel is None else round(rev_rel, 6),
                "gt_rows": len(gt_rows),
                "target_rows": len(target_rows),
                "gt_source": gt_source,
                "gt_filepath": gt["filepath"],
                "filepath": target["filepath"],
            })

    # Print warnings
    for w in warnings:
        print(w, file=sys.stderr)

    # Print & save results
    if not results:
        print("No comparisons were made. Check your filters or data.")
        sys.exit(0)

    # Sort results: by query number, then sel, then case, then index
    results.sort(key=lambda r: (query_sort_key(r["query"]), r["sel"], r["case"], r["index"]))

    # Write CSV
    import csv
    fieldnames = [
        "query", "sf", "sel", "case", "device", "index",
        "relaxed_recall", "strict_recall",
        "rev_err_abs", "rev_err_rel",  # Q19-only; blank for set-output queries
        "gt_rows", "target_rows",
        "gt_source", "gt_filepath", "filepath",
    ]

    os.makedirs(args.recall_dir, exist_ok=True)
    out_path = os.path.join(args.recall_dir, args.out)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nRecall report saved to: {out_path}")
    print(f"Total comparisons: {len(results)}")

    # GT cross-check summary
    gt_warnings = [w for w in warnings if "MISMATCH" in w]
    if gt_warnings:
        print(f"\n[WARN] CPU vs GPU ground-truth mismatches detected ({len(gt_warnings)}):")
        for w in gt_warnings:
            print(f"  {w}")
    else:
        print("\n[OK] CPU vs GPU ground-truth cross-check passed (all matching)")

    # Summary table to stdout. Extra trailing 'RevErr%' column is Q19-only
    # (printed for q19_start rows; blank everywhere else).
    has_sel = any(r["sel"] for r in results)
    rev_col = f" {'RevErr%':>8}"
    if has_sel:
        hdr = f"{'Query':<18} {'Sel':<5} {'Case':<22} {'Index':<26} {'Relaxed':>8} {'Strict':>8} {'GT':>6} {'Tgt':>6}{rev_col}"
    else:
        hdr = f"{'Query':<18} {'Case':<22} {'Index':<26} {'Relaxed':>8} {'Strict':>8} {'GT':>6} {'Tgt':>6}{rev_col}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for r in results:
        perfect_both = (r["relaxed_recall"] == 1.0 and r["strict_recall"] == 1.0)
        marker = " " if perfect_both else "!"
        rev_str = f" {100*r['rev_err_rel']:>7.3f}%" if r["rev_err_rel"] != "" else f" {'':>8}"
        if has_sel:
            print(
                f"{marker}{r['query']:<17} {r['sel']:<5} {r['case']:<22} {r['index']:<26} "
                f"{r['relaxed_recall']:>7.4f}  {r['strict_recall']:>7.4f}  {r['gt_rows']:>5}  {r['target_rows']:>5}{rev_str}"
            )
        else:
            print(
                f"{marker}{r['query']:<17} {r['case']:<22} {r['index']:<26} "
                f"{r['relaxed_recall']:>7.4f}  {r['strict_recall']:>7.4f}  {r['gt_rows']:>5}  {r['target_rows']:>5}{rev_str}"
            )
    print("-" * len(hdr))

    perfect = sum(1 for r in results if r["relaxed_recall"] == 1.0 and r["strict_recall"] == 1.0)
    print(f"\nPerfect recall (relaxed+strict): {perfect}/{len(results)}")
    imperfect = [r for r in results if r["relaxed_recall"] < 1.0 or r["strict_recall"] < 1.0]
    if imperfect:
        print(f"Imperfect recall ({len(imperfect)}):")
        for r in imperfect:
            sel_str = f" sel={r['sel']}" if r["sel"] else ""
            print(f"  {r['query']:<17}{sel_str} {r['case']:<22} {r['index']:<26} "
                  f"relaxed={r['relaxed_recall']:.4f}  strict={r['strict_recall']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate relaxed & strict recall for VSDS/vary-batch benchmark CSV outputs."
    )
    parser.add_argument(
        "--results_dir", default="./vsds",
        help="Root results directory to scan (default: ./vsds)."
    )
    parser.add_argument("--sf", default=None, help="Filter by scale factor (e.g. '1', '0.01')")
    parser.add_argument("--system", default=None, help="Filter by system name (e.g. 'dgx-spark-02')")
    parser.add_argument("--query", default=None, help="Filter by query name (e.g. 'q2_start')")
    parser.add_argument("--k", default=None, help="Filter by k value (e.g. '100')")
    parser.add_argument(
        "--out", default="recall_report.csv",
        help="Output CSV filename (default: recall_report.csv)"
    )
    parser.add_argument(
        "--recall_dir", default="./recall",
        help="Output directory for recall results (default: ./recall)"
    )
    parser.add_argument(
        "--batch_filter", default=None,
        help="Only include files with this batch size (e.g. '10000'). "
             "Use for vary-batch to only compare the superset batch."
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
