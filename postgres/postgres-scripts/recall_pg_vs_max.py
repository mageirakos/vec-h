#!/usr/bin/env python3
"""
pgvector ENN vs Maximus Flat recall.

For each pgvector ENN csv (`<pg_run_dir>/csv/enn/<query>.csv`), find all
Maximus Flat-family CSVs for that query at `--sf` under `<max_run_dir>`
(`maximus_q_<query>-i_{Flat,GPU,Flat}-...-sf_<sf>*.csv`) and compute
relaxed + strict recall (pgvector ENN is the ground truth).

CPU Flat (i=Flat, d_cpu-s_cpu-is_cpu) is the primary comparison and is
marked with `*` in the console summary. Maximus-internal cross-check
(CPU Flat vs GPU Flat variants) emits a warning if they disagree.

Usage:
  python recall_pg_vs_max.py <pg_run_dir> <max_run_dir> [--sf 1]
                             [--out recall_pg_vs_max.csv]
                             [--warn_threshold 0.99]

Manual run (inter-engine: pgvector ENN ground truth vs Maximus Flat family):
  cd /local/home/vmageirakos/projects/maxvs-temp/maxvec
  python postgres-scripts/recall_pg_vs_max.py \
    postgres-default/results/runs/11Apr26-default/run5 \
    /local/home/vmageirakos/projects/Maximus/results/runs/04Apr26-default/run1 \
    --sf 1 --also_read
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict


FLOAT_TRUNC_RE = re.compile(r"(\.\d{4})\d+")

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

FLAT_INDEXES = {"Flat", "GPU,Flat"}

# Display name mapping — explicit "raw" → "shown" so the same logical index
# prints identically across the three tables. Unmapped names pass through
# unchanged (so a new variant is visible rather than silently mangled).
# Keys with "(C+H)" / "(H)" are produced by _tag_index() before lookup.
INDEX_DISPLAY = {
    # ENN (main table + intra)
    "Flat":                              "Flat",
    "GPU,Flat":                          "GPU,Flat",
    # IVF — CPU + GPU + host-data variant
    "IVF1024,Flat":                      "IVF1024",
    "IVF4096,Flat":                      "IVF4096",
    "GPU,IVF1024,Flat":                  "GPU,IVF1024",
    "GPU,IVF4096,Flat":                  "GPU,IVF4096",
    "GPU,IVF1024(H),Flat":               "GPU,IVF1024(H)",
    "GPU,IVF4096(H),Flat":               "GPU,IVF4096(H)",
    # CAGRA — default + cache-graph+host variant
    "GPU,Cagra,64,32,NN_DESCENT":        "GPU,Cagra",
    "GPU,Cagra(C+H),64,32,NN_DESCENT":   "GPU,Cagra(C+H)",
    # pgvector intra
    "HNSW32":                            "HNSW32",
    "IVF1024":                           "IVF1024",
}


def _shorten_index(name: str) -> str:
    return INDEX_DISPLAY.get(name, name)

# Skip distance columns: metric differs between pgvector and Maximus/faiss
# (sign flip for IP etc.), so direct equality is meaningless.
DISTANCE_COLS = {"vs_distance", "avg_semantic_dist", "visual_distance", "semantic_distance"}

# Columns where Maximus returns empty/NaN and pgvector returns 0 — treat as equal.
ZERO_NORM_COLS = {
    "q18": {"similar_qty", "num_similar_items"},
    "q13": {"reviewdist"},
}


def parse_max_filename(fname: str):
    m = FILE_PATTERN.match(fname)
    return m.groupdict() if m else None


_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


def normalize_cell(s: str) -> str:
    s = s.strip() if s is not None else ""
    low = s.lower()
    if low in ("true", "false"):
        return low
    if low in ("", "nan", "none"):
        return ""
    # Canonicalize numerics so 1659.25 == 1659.250000: parse and re-emit
    # at fixed precision, then strip trailing zeros.
    if _NUM_RE.match(s):
        try:
            v = float(s)
            return f"{v:.4f}".rstrip("0").rstrip(".")
        except ValueError:
            pass
    return s


def load_pg_csv(path: str):
    """Return (cols, rows_as_list_of_dict)."""
    with open(path) as f:
        rdr = csv.reader(f)
        cols = [c.strip() for c in next(rdr)]
        rows = [dict(zip(cols, [c for c in r])) for r in rdr if r]
    return cols, rows


def load_max_csv(path: str):
    """Maximus csv: ', ' separator, type-annotated headers `col(dtype)`.
    Unquoted free-text fields may contain bare commas (always followed by a
    non-space char), so split with maxsplit=ncols-1 on ', '. Convert
    date32 integer-days values to 'YYYY-MM-DD' to match pgvector dates."""
    from datetime import date, timedelta
    epoch = date(1970, 1, 1)

    with open(path) as fh:
        header = fh.readline().rstrip("\n")
    cols = []
    types = {}
    for tok in header.split(","):
        tok = tok.strip()
        m = re.match(r"^(\w+)\((\w+)\)$", tok)
        if m:
            col, dtype = m.group(1), m.group(2)
        else:
            col, dtype = tok, "unknown"
        cols.append(col)
        types[col] = dtype
    ncols = len(cols)
    date_cols = [c for c, t in types.items() if t == "date32"]

    rows = []
    with open(path) as fh:
        fh.readline()
        for line in fh:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split(", ", ncols - 1)
            if len(parts) < ncols:
                continue
            row = dict(zip(cols, [p.strip() for p in parts]))
            for c in date_cols:
                v = row.get(c, "")
                if v and v.lstrip("-").isdigit():
                    row[c] = (epoch + timedelta(days=int(v))).isoformat()
            rows.append(row)
    return cols, rows


def rows_to_canonical(rows, cols, zero_norm=frozenset()):
    out = []
    for r in rows:
        parts = []
        for c in cols:
            v = normalize_cell(r.get(c, ""))
            if c in zero_norm and v in ("0", "0.0", ""):
                v = ""
            parts.append(v)
        out.append("|".join(parts))
    return out


def _zero_norm_for(query: str) -> set:
    out = set()
    for key, cols in ZERO_NORM_COLS.items():
        if key in query.lower():
            out |= cols
    return out


# Q19-only: revenue-error alongside the usual strict/relaxed recall.
def q19_revenue_error(gt_path, max_path):
    """Read the single revenue value from each CSV; return (abs_err, rel_err)."""
    def _read(p):
        with open(p) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return float(lines[1])  # header on line 0, value on line 1
    try:
        rev_exact = _read(gt_path)
        rev_got   = _read(max_path)
    except (IndexError, ValueError, OSError):
        return (None, None)
    abs_err = rev_got - rev_exact
    rel_err = abs(abs_err) / rev_exact if rev_exact else float("nan")
    return (abs_err, rel_err)


def compute_recall(gt_rows, target_rows):
    n_gt = len(gt_rows)
    if n_gt == 0:
        return (1.0, 1.0) if len(target_rows) == 0 else (0.0, 0.0)
    gt_c = Counter(gt_rows)
    tg_c = Counter(target_rows)
    match_rel = sum(min(c, tg_c.get(r, 0)) for r, c in gt_c.items())
    match_strict = sum(
        1 for i in range(min(n_gt, len(target_rows))) if gt_rows[i] == target_rows[i]
    )
    return match_rel / n_gt, match_strict / n_gt


def query_sort_key(q: str):
    m = re.match(r"q(\d+)", q)
    return (int(m.group(1)) if m else 999, q)


def case_label(info) -> str:
    return f"d_{info['device']}-s_{info['storage']}-is_{info['index_storage']}"


def is_cpu_flat(info) -> bool:
    return (info["index"] == "Flat" and info["device"] == "cpu"
            and info["storage"] == "cpu" and info["index_storage"] == "cpu")


def _is_case0_or_c3_flat(info) -> bool:
    """Keep only case 0 (CPU Flat, d_cpu-s_cpu-is_cpu) and case 3
    (all-GPU Flat, GPU,Flat / d_gpu-s_gpu-is_gpu)."""
    if info["index"] == "Flat" and info["device"] == "cpu" \
            and info["storage"] == "cpu" and info["index_storage"] == "cpu":
        return True
    if info["index"] == "GPU,Flat" and info["device"] == "gpu" \
            and info["storage"] == "gpu" and info["index_storage"] == "gpu":
        return True
    return False


def discover_max_flat(max_run_dir: str, sf: str):
    by_query = defaultdict(list)
    for dp, _, fnames in os.walk(max_run_dir):
        for fn in fnames:
            if not fn.endswith(".csv"):
                continue
            info = parse_max_filename(fn)
            if info is None:
                continue
            if not _is_case0_or_c3_flat(info):
                continue
            if sf and info["sf"] != sf:
                continue
            info["filepath"] = os.path.join(dp, fn)
            info["filename"] = fn
            by_query[info["query"]].append(info)
    return by_query


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pg_run_dir", help="pgvector run dir containing csv/enn/")
    p.add_argument("max_run_dir", help="Maximus run dir (searched recursively)")
    p.add_argument("--sf", default="1")
    p.add_argument("--out", default="recall_pg_vs_max.csv")
    p.add_argument("--warn_threshold", type=float, default=0.99)
    p.add_argument(
        "--also_read", action="store_true",
        help="After the main inter-engine table, also print two intra-engine "
             "ANN tables read from pre-computed CSVs: "
             "<max_run_dir>/recall/recall_report.csv (Maximus ANN vs Maximus CPU Flat) and "
             "<pg_run_dir>/recall/recall_intra.csv (pgvector ANN vs pgvector ENN).",
    )
    p.add_argument("--also_read_max_csv", default=None,
                   help="Override path for Maximus intra recall CSV (default: <max_run_dir>/recall/recall_report.csv)")
    p.add_argument("--also_read_pg_csv", default=None,
                   help="Override path for pgvector intra recall CSV (default: <pg_run_dir>/recall/recall_intra.csv)")
    args = p.parse_args()

    enn_dir = os.path.join(args.pg_run_dir, "csv", "enn")
    if not os.path.isdir(enn_dir):
        print(f"ERROR: no pg ENN dir at {enn_dir}", file=sys.stderr)
        sys.exit(1)

    max_flat = discover_max_flat(args.max_run_dir, args.sf)
    if not max_flat:
        print(f"ERROR: no Maximus Flat csvs under {args.max_run_dir} (sf={args.sf})", file=sys.stderr)
        sys.exit(1)

    results = []
    warnings_out = []

    for pg_fname in sorted(os.listdir(enn_dir)):
        if not pg_fname.endswith(".csv"):
            continue
        query = pg_fname[:-4]
        pg_path = os.path.join(enn_dir, pg_fname)
        max_files = max_flat.get(query, [])
        if not max_files:
            warnings_out.append(f"[WARN] no Maximus Flat csv for query={query}")
            continue

        pg_cols, pg_rows = load_pg_csv(pg_path)

        # Per-query recall for each Maximus variant. Columns = intersection,
        # sorted alphabetically for a stable canonical row representation.
        for info in max_files:
            try:
                max_cols, max_rows = load_max_csv(info["filepath"])
            except Exception as e:
                warnings_out.append(f"[WARN] failed to parse {info['filename']}: {e}")
                continue
            common = sorted((set(pg_cols) & set(max_cols)) - DISTANCE_COLS)
            if not common:
                warnings_out.append(f"[WARN] no common columns for {query} vs {info['filename']}")
                continue
            zero_norm = _zero_norm_for(query)
            gt = rows_to_canonical(pg_rows, common, zero_norm)
            tgt = rows_to_canonical(max_rows, common, zero_norm)
            rel, strict = compute_recall(gt, tgt)
            # q11_end aux: recall with duplicate_partkey dropped — that column is
            # a tie-breaking artifact (equidistant representative image per part).
            if query == "q11_end" and "duplicate_partkey" in common:
                common_nt = [c for c in common if c != "duplicate_partkey"]
                gt_nt  = rows_to_canonical(pg_rows, common_nt, zero_norm)
                tgt_nt = rows_to_canonical(max_rows, common_nt, zero_norm)
                q11_rel, q11_strict = compute_recall(gt_nt, tgt_nt)
            else:
                q11_rel = q11_strict = None
            primary = is_cpu_flat(info)
            rev_abs, rev_rel = (q19_revenue_error(pg_path, info["filepath"])
                                if query == "q19_start" else (None, None))
            results.append({
                "query": query,
                "sf": args.sf,
                "max_index": info["index"],
                "max_case": case_label(info),
                "primary": "yes" if primary else "",
                "relaxed_recall": round(rel, 6),
                "strict_recall": round(strict, 6),
                "rev_err_abs": "" if rev_abs is None else round(rev_abs, 4),
                "rev_err_rel": "" if rev_rel is None else round(rev_rel, 6),
                "q11_notie_relaxed": "" if q11_rel is None else round(q11_rel, 6),
                "q11_notie_strict":  "" if q11_strict is None else round(q11_strict, 6),
                "gt_rows": len(gt),
                "target_rows": len(tgt),
                "pg_filepath": pg_path,
                "max_filepath": info["filepath"],
            })
            if rel < args.warn_threshold or strict < args.warn_threshold:
                warnings_out.append(
                    f"[WARN] low recall: {query} {info['index']} {case_label(info)} "
                    f"relaxed={rel:.4f} strict={strict:.4f}"
                )

        # Maximus-internal cross-check: CPU Flat vs each GPU Flat variant.
        cpu_rel = next(
            (results[-len(max_files) + i]["relaxed_recall"]
             for i, info in enumerate(max_files) if is_cpu_flat(info)),
            None,
        )
        if cpu_rel is not None:
            for r in results[-len(max_files):]:
                if r["primary"] == "yes":
                    continue
                if abs(r["relaxed_recall"] - cpu_rel) > 0.01:
                    warnings_out.append(
                        f"[WARN] Maximus CPU Flat vs {r['max_index']} {r['max_case']} disagree "
                        f"for {query}: {cpu_rel:.4f} vs {r['relaxed_recall']:.4f}"
                    )

    if not results:
        print("No comparisons made.")
        sys.exit(0)

    results.sort(key=lambda r: (query_sort_key(r["query"]), r["max_index"], r["max_case"]))

    out_dir = os.path.join(args.pg_run_dir, "recall")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out)
    fieldnames = [
        "query", "sf", "max_index", "max_case", "primary",
        "relaxed_recall", "strict_recall",
        "rev_err_abs", "rev_err_rel",                  # q19_start-only
        "q11_notie_relaxed", "q11_notie_strict",       # q11_end-only: duplicate_partkey-dropped recall (tie-breaking aux)
        "gt_rows", "target_rows",
        "pg_filepath", "max_filepath",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    for w in warnings_out:
        print(w, file=sys.stderr)

    hdr = f"{'Query':<18} {'Max index':<18} {'Max case':<24} {'Relaxed':>8} {'Strict':>8} {'GT':>6} {'Tgt':>6} {'RevErr%':>8} {'q11NoTieRel':>11} {'q11NoTieStr':>11}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for r in results:
        marker = "*" if r["primary"] == "yes" else (
            " " if (r["relaxed_recall"] == 1.0 and r["strict_recall"] == 1.0) else "!"
        )
        rev_str = f" {100*r['rev_err_rel']:>7.3f}%" if r["rev_err_rel"] != "" else f" {'':>8}"
        q11_str = (f" {r['q11_notie_relaxed']:>11.4f} {r['q11_notie_strict']:>11.4f}"
                   if r["q11_notie_relaxed"] != "" else f" {'':>11} {'':>11}")
        print(f"{marker}{r['query']:<17} {_shorten_index(r['max_index']):<18} {r['max_case']:<24} "
              f"{r['relaxed_recall']:>7.4f}  {r['strict_recall']:>7.4f}  "
              f"{r['gt_rows']:>5}  {r['target_rows']:>5}{rev_str}{q11_str}")
    print("-" * len(hdr))
    if any(r["q11_notie_relaxed"] != "" for r in results):
        print("  q11NoTieRel/q11NoTieStr: recall with duplicate_partkey dropped — that column")
        print("  is a tie-breaking artifact (equidistant representative image per part).")
    perfect = sum(1 for r in results if r["relaxed_recall"] == 1.0 and r["strict_recall"] == 1.0)
    print(f"\nPerfect recall (relaxed+strict): {perfect}/{len(results)}")
    print(f"Recall report saved to: {out_path}")

    if args.also_read:
        queries_seen = {r["query"] for r in results}
        max_csv = args.also_read_max_csv or os.path.join(args.max_run_dir, "recall", "recall_report.csv")
        pg_csv  = args.also_read_pg_csv  or os.path.join(args.pg_run_dir,  "recall", "recall_intra.csv")
        _print_intra_table(
            "Maximus intra — ANN vs Maximus CPU Flat ENN (ground truth)",
            max_csv, queries_seen, args.sf,
            exclude_indexes={"Flat", "GPU,Flat"},
            case_col="case",
            case_filter={"d_cpu-s_cpu-is_cpu", "d_gpu-s_gpu-is_gpu"},  # case 0 + case 3 only
        )
        _print_intra_table(
            "pgvector intra — ANN (HNSW32, IVF1024) vs pgvector ENN (ground truth)",
            pg_csv, queries_seen, args.sf,
            exclude_indexes=set(),
            case_col=None,
            case_filter=None,
        )


def _tag_index(index: str, filepath: str) -> str:
    """Disambiguate same-named indexes built with different params, detected
    from filename suffixes set by run_vech.sh:
      -cagra_ch  → Cagra(C+H)   (cache_graph=1, data_on_gpu=0)
      -ivf_h     → IVF(H)       (IVF with host data placement variant)
    """
    fp = filepath or ""
    if "-cagra_ch" in fp and "Cagra" in index:
        index = index.replace("Cagra", "Cagra(C+H)")
    elif "-ivf_h" in fp and "IVF" in index:
        index = re.sub(r"(IVF\d+)", r"\1(H)", index)
    return _shorten_index(index)


def _print_intra_table(title, csv_path, queries_filter, sf, exclude_indexes, case_col, case_filter):
    """Render a pre-computed intra-engine recall CSV as a table. Skips quietly
    if the file is missing — this is an optional convenience view."""
    if not os.path.isfile(csv_path):
        print(f"\n[also_read] skip: {csv_path} not found", file=sys.stderr)
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows
            if r.get("query") in queries_filter
            and (not sf or r.get("sf", "") in (sf, ""))
            and r.get("index", "") not in exclude_indexes
            and (case_filter is None or r.get(case_col or "", "") in case_filter)]
    if not rows:
        print(f"\n[also_read] {title}: no matching rows in {csv_path}")
        return
    for r in rows:
        r["_display_index"] = _tag_index(r.get("index", ""), r.get("filepath", ""))
    rows.sort(key=lambda r: (query_sort_key(r["query"]), r["_display_index"], r.get(case_col or "", "")))
    if case_col:
        hdr = f"{'Query':<18} {'Index':<26} {'Case':<22} {'Relaxed':>8} {'Strict':>8} {'RevErr%':>8}"
    else:
        hdr = f"{'Query':<18} {'Index':<26} {'Relaxed':>8} {'Strict':>8} {'RevErr%':>8}"
    print(f"\n{title}")
    print(f"(source: {csv_path})")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        rel = float(r["relaxed_recall"]); strict = float(r["strict_recall"])
        marker = " " if (rel == 1.0 and strict == 1.0) else "!"
        rev = r.get("rev_err_rel", "")
        rev_str = f" {100*float(rev):>7.3f}%" if rev not in ("", None) else f" {'':>8}"
        idx = r.get("_display_index") or r.get("index", "")
        if case_col:
            print(f"{marker}{r['query']:<17} {idx:<26} {r.get(case_col,''):<22} "
                  f"{rel:>7.4f}  {strict:>7.4f}{rev_str}")
        else:
            print(f"{marker}{r['query']:<17} {idx:<26} "
                  f"{rel:>7.4f}  {strict:>7.4f}{rev_str}")
    print("-" * len(hdr))


if __name__ == "__main__":
    main()
