#!/usr/bin/env python3
"""
Intra-pgvector recall: ENN baseline vs ANN indexes (HNSW32, IVF1024).

Layout expected:
  <run_dir>/csv/enn/<query>.csv
  <run_dir>/csv/HNSW32/<query>.csv
  <run_dir>/csv/IVF1024/<query>.csv

Usage:
  python recall_pg_intra.py <run_dir> [--sf 1] [--out recall_intra.csv]
                            [--warn_threshold 0.99]

Manual run (pgvector intra-engine):
  cd /local/home/vmageirakos/projects/maxvs-temp/maxvec
  python postgres-scripts/recall_pg_intra.py \
    postgres-default/results/runs/11Apr26-default/run5 --sf 1

Style mirrors Maximus/results/recall_vech.py (relaxed + strict recall,
summary table, CSV output).
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter


FLOAT_TRUNC_RE = re.compile(r"(\.\d{4})\d+")


def normalize_row(row_str: str) -> str:
    return FLOAT_TRUNC_RE.sub(r"\1", row_str).strip()


def load_rows(filepath: str) -> list[str]:
    with open(filepath) as f:
        lines = f.readlines()
    if len(lines) <= 1:
        return []
    return [normalize_row(line) for line in lines[1:] if line.strip()]


# Q11-only auxiliary: `duplicate_partkey` is the representative image per
# part chosen by the vector search. When multiple images are equidistant
# to the query (verified at 4-decimal precision for all observed
# disagreements), ENN / HNSW / IVF may pick different but equally-valid
# representatives. This aux recall drops that column so only the actual
# query answer — (ps_partkey, value) and related columns — is compared.
def q11_notie_recall(gt_path: str, target_path: str):
    def _load(p):
        with open(p) as f:
            r = csv.reader(f)
            header = next(r)
            keep = [i for i, c in enumerate(header) if c != "duplicate_partkey"]
            return [normalize_row(",".join(row[i] for i in keep)) for row in r if row]
    gt, tg = _load(gt_path), _load(target_path)
    rel, strict, _, _ = compute_recall(gt, tg)
    return rel, strict


# Q19-only: revenue-error alongside strict/relaxed recall (1-row scalar SUM).
def q19_revenue_error(gt_rows, target_rows):
    try:
        rev_exact = float(gt_rows[0])
        rev_got   = float(target_rows[0])
    except (IndexError, ValueError):
        return (None, None)
    abs_err = rev_got - rev_exact
    rel_err = abs(abs_err) / rev_exact if rev_exact else float("nan")
    return (abs_err, rel_err)


def compute_recall(gt_rows, target_rows):
    n_gt = len(gt_rows)
    if n_gt == 0:
        return (1.0, 1.0, 0, 0) if len(target_rows) == 0 else (0.0, 0.0, 0, 0)
    gt_c = Counter(gt_rows)
    tg_c = Counter(target_rows)
    match_rel = sum(min(c, tg_c.get(r, 0)) for r, c in gt_c.items())
    match_strict = sum(
        1 for i in range(min(n_gt, len(target_rows))) if gt_rows[i] == target_rows[i]
    )
    return match_rel / n_gt, match_strict / n_gt, match_rel, match_strict


def query_sort_key(q: str):
    m = re.match(r"q(\d+)", q)
    return (int(m.group(1)) if m else 999, q)


ANN_INDEXES = ["HNSW32", "IVF1024"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="pgvector run dir (contains csv/enn, csv/HNSW32, ...)")
    parser.add_argument("--sf", default="")
    parser.add_argument("--out", default="recall_intra.csv")
    parser.add_argument("--warn_threshold", type=float, default=0.99)
    args = parser.parse_args()

    csv_root = os.path.join(args.run_dir, "csv")
    enn_dir = os.path.join(csv_root, "enn")
    if not os.path.isdir(enn_dir):
        print(f"ERROR: no ENN dir at {enn_dir}", file=sys.stderr)
        sys.exit(1)

    enn_files = sorted(
        f for f in os.listdir(enn_dir) if f.endswith(".csv")
    )
    if not enn_files:
        print(f"ERROR: no ENN csvs in {enn_dir}", file=sys.stderr)
        sys.exit(1)

    results = []
    warnings_out = []

    for fname in enn_files:
        query = fname[:-4]
        gt_path = os.path.join(enn_dir, fname)
        gt_rows = load_rows(gt_path)

        for idx in ANN_INDEXES:
            tgt_path = os.path.join(csv_root, idx, fname)
            if not os.path.isfile(tgt_path):
                continue
            tgt_rows = load_rows(tgt_path)
            rel, strict, _, _ = compute_recall(gt_rows, tgt_rows)
            rev_abs, rev_rel = (q19_revenue_error(gt_rows, tgt_rows)
                                if query == "q19_start" else (None, None))
            q11_rel, q11_strict = (q11_notie_recall(gt_path, tgt_path)
                                   if query == "q11_end" else (None, None))
            results.append({
                "query": query,
                "sf": args.sf,
                "index": idx,
                "relaxed_recall": round(rel, 6),
                "strict_recall": round(strict, 6),
                "rev_err_abs": "" if rev_abs is None else round(rev_abs, 4),
                "rev_err_rel": "" if rev_rel is None else round(rev_rel, 6),
                "q11_notie_relaxed": "" if q11_rel is None else round(q11_rel, 6),
                "q11_notie_strict":  "" if q11_strict is None else round(q11_strict, 6),
                "gt_rows": len(gt_rows),
                "target_rows": len(tgt_rows),
                "gt_filepath": gt_path,
                "filepath": tgt_path,
            })
            if rel < args.warn_threshold or strict < args.warn_threshold:
                warnings_out.append(
                    f"[WARN] low recall: {query} {idx} relaxed={rel:.4f} strict={strict:.4f}"
                )

    if not results:
        print("No ANN targets found alongside ENN — nothing to compare.")
        sys.exit(0)

    results.sort(key=lambda r: (query_sort_key(r["query"]), r["index"]))

    out_dir = os.path.join(args.run_dir, "recall")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out)
    fieldnames = [
        "query", "sf", "index", "relaxed_recall", "strict_recall",
        "rev_err_abs", "rev_err_rel",                   # q19_start-only
        "q11_notie_relaxed", "q11_notie_strict",        # q11_end-only: recall with duplicate_partkey dropped (tie-breaking aux)
        "gt_rows", "target_rows", "gt_filepath", "filepath",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    for w in warnings_out:
        print(w, file=sys.stderr)

    hdr = f"{'Query':<18} {'Index':<10} {'Relaxed':>8} {'Strict':>8} {'GT':>6} {'Tgt':>6} {'RevErr%':>8} {'q11NoTieRel':>11} {'q11NoTieStr':>11}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for r in results:
        marker = " " if (r["relaxed_recall"] == 1.0 and r["strict_recall"] == 1.0) else "!"
        rev_str  = f" {100*r['rev_err_rel']:>7.3f}%" if r["rev_err_rel"] != "" else f" {'':>8}"
        q11_str  = (f" {r['q11_notie_relaxed']:>11.4f} {r['q11_notie_strict']:>11.4f}"
                    if r["q11_notie_relaxed"] != "" else f" {'':>11} {'':>11}")
        print(f"{marker}{r['query']:<17} {r['index']:<10} "
              f"{r['relaxed_recall']:>7.4f}  {r['strict_recall']:>7.4f}  "
              f"{r['gt_rows']:>5}  {r['target_rows']:>5}{rev_str}{q11_str}")
    print("-" * len(hdr))
    if any(r["q11_notie_relaxed"] != "" for r in results):
        print("  q11NoTieRel/q11NoTieStr: recall with duplicate_partkey dropped — that column")
        print("  is a tie-breaking artifact (equidistant representative image per part).")
    perfect = sum(1 for r in results if r["relaxed_recall"] == 1.0 and r["strict_recall"] == 1.0)
    print(f"\nPerfect recall (relaxed+strict): {perfect}/{len(results)}")
    print(f"Recall report saved to: {out_path}")


if __name__ == "__main__":
    main()
