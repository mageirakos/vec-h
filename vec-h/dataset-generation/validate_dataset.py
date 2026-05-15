"""
validate_dataset.py — Sanity-check the final VECH + TPC-H dataset.

Usage:
    python validate_dataset.py [--data-dir ./data]
"""

import argparse
import sys
from pathlib import Path

import polars as pl


# ── helpers ──────────────────────────────────────────────────────────────────

RESET = "\033[0m"
GREEN = "\033[32m"
RED   = "\033[31m"
YELLOW = "\033[33m"
BOLD  = "\033[1m"


def section(title: str) -> None:
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")


_warnings: list[str] = []
_failures: list[str] = []


def ok(msg: str) -> None:
    print(f"  {GREEN}PASS{RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}WARN{RESET}  {msg}")
    _warnings.append(msg)


def fail(msg: str) -> None:
    print(f"  {RED}FAIL{RESET}  {msg}")
    _failures.append(msg)


def info(msg: str) -> None:
    print(f"        {msg}")


# ── file existence ────────────────────────────────────────────────────────────

def check_files(vech_parquet: Path, vech_csv: Path, tpch: Path) -> None:
    section("FILE EXISTENCE")

    vech_tables = ["reviews", "reviews_queries", "images", "images_queries"]
    tpch_tables = ["customer", "part", "orders", "lineitem",
                   "supplier", "nation", "region", "partsupp"]

    for t in vech_tables:
        pq = vech_parquet / f"{t}.parquet"
        csv = vech_csv / f"{t}.csv"
        for p in (pq, csv):
            label = p.name
            if p.exists():
                ok(label)
            else:
                fail(f"Missing: {p}")

    for t in tpch_tables:
        pq = tpch / f"{t}.parquet"
        if pq.exists():
            ok(pq.name)
        else:
            fail(f"Missing: {pq}")


# ── load tables ───────────────────────────────────────────────────────────────

def load_tables(vech_parquet: Path, vech_csv: Path, tpch: Path) -> dict[str, pl.DataFrame]:
    tables: dict[str, pl.DataFrame] = {}

    vech = ["reviews", "reviews_queries", "images", "images_queries"]
    tpch_names = ["customer", "part", "orders", "lineitem",
                  "supplier", "nation", "region", "partsupp"]

    for t in vech:
        pq = vech_parquet / f"{t}.parquet"
        csv = vech_csv / f"{t}.csv"
        if pq.exists():
            tables[f"vech/{t}"] = pl.read_parquet(pq)
        if csv.exists():
            tables[f"vech_csv/{t}"] = pl.read_csv(csv)

    for t in tpch_names:
        pq = tpch / f"{t}.parquet"
        if pq.exists():
            tables[f"tpch/{t}"] = pl.read_parquet(pq)

    return tables


# ── row counts ────────────────────────────────────────────────────────────────

def check_row_counts(tables: dict[str, pl.DataFrame]) -> dict[str, int]:
    section("ROW COUNTS")
    counts: dict[str, int] = {}

    order = (
        ["vech/reviews", "vech/reviews_queries",
         "vech/images",  "vech/images_queries"]
        + [f"tpch/{t}" for t in ["customer", "part", "orders", "lineitem",
                                  "supplier", "nation", "region", "partsupp"]]
    )
    for key in order:
        if key in tables:
            n = len(tables[key])
            counts[key] = n
            info(f"{key:<40}  {n:>8,} rows")

    # cross-table summaries
    rv_data  = counts.get("vech/reviews", 0)
    rv_query = counts.get("vech/reviews_queries", 0)
    img_data = counts.get("vech/images", 0)
    img_query= counts.get("vech/images_queries", 0)
    print()
    info(f"reviews:  {rv_data:,} data  +  {rv_query:,} queries  =  {rv_data + rv_query:,} total")
    info(f"images:   {img_data:,} data  +  {img_query:,} queries  =  {img_data + img_query:,} total")
    return counts


# ── csv vs parquet consistency ────────────────────────────────────────────────

def check_csv_parity(tables: dict[str, pl.DataFrame], counts: dict[str, int]) -> None:
    section("CSV vs PARQUET ROW-COUNT CONSISTENCY")
    for key in ["reviews", "reviews_queries", "images", "images_queries"]:
        pq_n  = counts.get(f"vech/{key}")
        csv_n = len(tables[f"vech_csv/{key}"]) if f"vech_csv/{key}" in tables else None
        if pq_n is None or csv_n is None:
            warn(f"Cannot compare {key}: one format missing")
            continue
        if pq_n == csv_n:
            ok(f"{key}: parquet={pq_n:,}  csv={csv_n:,}  match")
        else:
            fail(f"{key}: parquet={pq_n:,}  csv={csv_n:,}  MISMATCH")


# ── schema ────────────────────────────────────────────────────────────────────

EXPECTED_COLS = {
    "reviews": [
        "rv_reviewkey", "rv_rating", "rv_helpful_vote",
        "rv_title", "rv_text", "rv_embedding", "rv_partkey", "rv_custkey",
    ],
    "reviews_queries": [
        "rv_reviewkey_queries", "rv_rating_queries", "rv_helpful_vote_queries",
        "rv_title_queries", "rv_text_queries", "rv_embedding_queries",
        "rv_partkey_queries", "rv_custkey_queries",
    ],
    "images": ["i_imagekey", "i_image_url", "i_variant", "i_embedding", "i_partkey"],
    "images_queries": [
        "i_imagekey_queries", "i_image_url_queries", "i_variant_queries",
        "i_embedding_queries", "i_partkey_queries",
    ],
}


def check_schema(tables: dict[str, pl.DataFrame]) -> None:
    section("SCHEMA")
    for t, expected in EXPECTED_COLS.items():
        key = f"vech/{t}"
        if key not in tables:
            warn(f"{t}: table not loaded, skipping")
            continue
        df = tables[key]
        actual = set(df.columns)
        missing = set(expected) - actual
        extra   = actual - set(expected)
        info(f"{t}: {df.columns}")
        if missing:
            fail(f"{t}: missing columns {missing}")
        elif extra:
            warn(f"{t}: unexpected extra columns {extra}")
        else:
            ok(f"{t}: all expected columns present")


# ── nulls ─────────────────────────────────────────────────────────────────────

def check_nulls(tables: dict[str, pl.DataFrame]) -> None:
    section("NULL / NaN COUNTS  (vech tables only)")
    for t in ["reviews", "reviews_queries", "images", "images_queries"]:
        key = f"vech/{t}"
        if key not in tables:
            continue
        df = tables[key]
        null_counts = df.null_count().row(0)  # one value per column
        print(f"\n  [{t}]")
        any_nulls = False
        for col, n in zip(df.columns, null_counts):
            if n > 0:
                fail(f"    {col}: {n:,} nulls")
                any_nulls = True
            else:
                info(f"    {col}: 0 nulls")
        if not any_nulls:
            ok(f"  {t}: no nulls in any column")


# ── duplicates ────────────────────────────────────────────────────────────────

def check_duplicates(tables: dict[str, pl.DataFrame]) -> None:
    section("DUPLICATE CHECKS")

    # Primary key uniqueness
    for pk, table in [("rv_reviewkey", "vech/reviews"),
                      ("rv_reviewkey_queries", "vech/reviews_queries"),
                      ("i_imagekey", "vech/images"),
                      ("i_imagekey_queries", "vech/images_queries")]:
        if table not in tables:
            continue
        df = tables[table]
        n = len(df)
        nu = df[pk].n_unique()
        if n == nu:
            ok(f"{table} {pk}: all {n:,} values unique")
        else:
            fail(f"{table} {pk}: {n - nu:,} duplicate keys (total={n:,}, unique={nu:,})")

    # Full-row duplicates
    print()
    for key in ["vech/reviews", "vech/reviews_queries", "vech/images", "vech/images_queries"]:
        if key not in tables:
            continue
        df = tables[key]
        # exclude embedding cols which are unhashable in some polars builds
        non_emb = [c for c in df.columns if "embedding" not in c]
        n = len(df)
        nu = df.select(non_emb).n_unique()
        dups = n - nu
        if dups == 0:
            ok(f"{key}: no full-row duplicates (excluding embeddings)")
        else:
            warn(f"{key}: {dups:,} duplicate rows (excluding embeddings)")

    # Near-duplicate check for reviews: (title, text, rating)
    print()
    for key, title_col, text_col, rating_col in [
        ("vech/reviews",         "rv_title",         "rv_text",         "rv_rating"),
        ("vech/reviews_queries", "rv_title_queries",  "rv_text_queries",  "rv_rating_queries"),
    ]:
        if key not in tables:
            continue
        df = tables[key]
        cols = [title_col, text_col, rating_col]
        if not all(c in df.columns for c in cols):
            continue
        n = len(df)
        nu = df.select(cols).n_unique()
        dups = n - nu
        if dups == 0:
            ok(f"{key}: no near-duplicate (title+text+rating) rows")
        else:
            warn(f"{key}: {dups:,} near-duplicate (title+text+rating) rows")

    # Embedding duplicates
    print()
    for emb_col, key in [
        ("rv_embedding",         "vech/reviews"),
        ("rv_embedding_queries", "vech/reviews_queries"),
        ("i_embedding",          "vech/images"),
        ("i_embedding_queries",  "vech/images_queries"),
    ]:
        if key not in tables:
            continue
        df = tables[key]
        if emb_col not in df.columns:
            continue
        n = len(df)
        try:
            nu = df[emb_col].n_unique()
            dups = n - nu
            if dups == 0:
                ok(f"{key} {emb_col}: all {n:,} embeddings unique")
            else:
                warn(f"{key} {emb_col}: {dups:,} duplicate embedding vectors ({dups/n*100:.2f}%)")
        except Exception as e:
            warn(f"{key} {emb_col}: could not check embedding uniqueness ({e})")


# ── foreign key integrity ─────────────────────────────────────────────────────

def check_foreign_keys(tables: dict[str, pl.DataFrame]) -> None:
    section("FOREIGN KEY INTEGRITY  (new tables → TPC-H)")

    part_keys = (
        tables["tpch/part"]["p_partkey"].to_list()
        if "tpch/part" in tables else None
    )
    cust_keys = (
        tables["tpch/customer"]["c_custkey"].to_list()
        if "tpch/customer" in tables else None
    )

    checks = [
        ("vech/reviews",         "rv_partkey",          "part.p_partkey",     part_keys),
        ("vech/reviews",         "rv_custkey",          "customer.c_custkey", cust_keys),
        ("vech/reviews_queries", "rv_partkey_queries",  "part.p_partkey",     part_keys),
        ("vech/reviews_queries", "rv_custkey_queries",  "customer.c_custkey", cust_keys),
        ("vech/images",          "i_partkey",           "part.p_partkey",     part_keys),
        ("vech/images_queries",  "i_partkey_queries",   "part.p_partkey",     part_keys),
    ]

    for table_key, fk_col, ref_label, ref_keys in checks:
        if table_key not in tables:
            continue
        df = tables[table_key]
        if fk_col not in df.columns:
            warn(f"{table_key}.{fk_col}: column not found")
            continue
        if ref_keys is None:
            warn(f"{table_key}.{fk_col} → {ref_label}: reference table not loaded")
            continue
        ref_set = set(ref_keys)
        fk_vals = df[fk_col]
        n_total = len(fk_vals)
        n_match = fk_vals.is_in(ref_set).sum()
        n_miss  = n_total - n_match
        pct = 100 * n_match / n_total if n_total > 0 else 0.0
        msg = (
            f"{table_key}.{fk_col} → {ref_label}: "
            f"{n_match:,}/{n_total:,} matched ({pct:.1f}%),  {n_miss:,} unmatched"
        )
        if n_miss == 0:
            ok(msg)
        else:
            fail(msg)


# ── embeddings ────────────────────────────────────────────────────────────────

def check_embeddings(tables: dict[str, pl.DataFrame]) -> None:
    section("EMBEDDING SHAPE & STATS")

    specs = [
        ("vech/reviews",         "rv_embedding",         1024),
        ("vech/reviews_queries", "rv_embedding_queries",  1024),
        ("vech/images",          "i_embedding",          1152),
        ("vech/images_queries",  "i_embedding_queries",  1152),
    ]

    for table_key, emb_col, expected_dim in specs:
        if table_key not in tables:
            continue
        df = tables[table_key]
        if emb_col not in df.columns:
            warn(f"{table_key}: embedding column '{emb_col}' missing")
            continue

        s = df[emb_col]
        print(f"\n  [{table_key}  {emb_col}]")
        info(f"dtype:          {s.dtype}")

        # lengths
        lengths = s.list.len()
        min_len, max_len = lengths.min(), lengths.max()
        info(f"embedding dims: min={min_len}  max={max_len}  (expected {expected_dim})")
        if min_len == max_len == expected_dim:
            ok(f"all embeddings have correct dimension {expected_dim}")
        else:
            fail(f"dimension mismatch: expected {expected_dim}, got min={min_len} max={max_len}")

        # value stats on a sample to avoid materializing everything
        sample = s.head(min(200, len(s)))
        first_vals = sample.list.get(0).cast(pl.Float64)
        info(f"first-element stats (sample={len(sample)}):  "
             f"mean={first_vals.mean():.4f}  "
             f"min={first_vals.min():.4f}  "
             f"max={first_vals.max():.4f}")

        # L2 norm of first few rows
        arr = sample.list.to_array(expected_dim)
        norms = pl.Series(
            [float(sum(x**2 for x in row)**0.5) for row in arr.to_list()]
        )
        info(f"L2 norm (sample):  mean={norms.mean():.4f}  "
             f"min={norms.min():.4f}  max={norms.max():.4f}")
        if norms.mean() < 0.01:
            fail(f"mean L2 norm near zero — embeddings may be all zeros")
        else:
            ok(f"embeddings look non-trivial (mean L2 norm={norms.mean():.4f})")


# ── value ranges ──────────────────────────────────────────────────────────────

def check_value_ranges(tables: dict[str, pl.DataFrame]) -> None:
    section("VALUE RANGE CHECKS")

    # rv_rating
    for key, col in [("vech/reviews", "rv_rating"), ("vech/reviews_queries", "rv_rating_queries")]:
        if key not in tables or col not in tables[key].columns:
            continue
        s = tables[key][col]
        info(f"{key} {col}:  min={s.min()}  max={s.max()}  mean={s.mean():.2f}")
        if s.min() < 1 or s.max() > 5:
            warn(f"{key} {col}: values outside expected [1, 5] range")
        else:
            ok(f"{key} {col}: values in [1, 5]")

    # rv_helpful_vote
    for key, col in [("vech/reviews", "rv_helpful_vote"),
                     ("vech/reviews_queries", "rv_helpful_vote_queries")]:
        if key not in tables or col not in tables[key].columns:
            continue
        s = tables[key][col]
        info(f"{key} {col}:  min={s.min()}  max={s.max()}  mean={s.mean():.2f}")
        if s.min() < 0:
            warn(f"{key} {col}: negative helpful_vote values found")
        else:
            ok(f"{key} {col}: all values >= 0")

    # i_variant unique values
    for key, col in [("vech/images", "i_variant"), ("vech/images_queries", "i_variant_queries")]:
        if key not in tables or col not in tables[key].columns:
            continue
        vc = tables[key][col].value_counts().sort("count", descending=True)
        info(f"{key} {col} unique values ({len(vc)}):")
        for row in vc.iter_rows(named=True):
            info(f"    {row[col]!r:<20}  {row['count']:>6,}")


# ── summary ───────────────────────────────────────────────────────────────────

def print_summary() -> None:
    section("SUMMARY")
    n_warn = len(_warnings)
    n_fail = len(_failures)
    info(f"Failures:  {n_fail}")
    info(f"Warnings:  {n_warn}")
    if n_fail == 0 and n_warn == 0:
        print(f"\n  {GREEN}{BOLD}All checks passed.{RESET}")
    else:
        if n_fail > 0:
            print(f"\n  {RED}{BOLD}FAILURES:{RESET}")
            for f in _failures:
                print(f"    {RED}•{RESET} {f}")
        if n_warn > 0:
            print(f"\n  {YELLOW}{BOLD}WARNINGS:{RESET}")
            for w in _warnings:
                print(f"    {YELLOW}•{RESET} {w}")


# ── main ──────────────────────────────────────────────────────────────────────

def report_scale_factor(tables: dict[str, pl.DataFrame]) -> None:
    """Derive the TPC-H scale factor from table cardinalities (not from vech —
    vech row counts are driven by the Amazon dataset, not SF)."""
    for key, base in [("tpch/part", 200_000), ("tpch/customer", 150_000)]:
        if key in tables:
            n = len(tables[key])
            info(f"Derived SF from {key}: {n:,} / {base:,} = {n / base:g}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate the VECH + TPC-H dataset.")
    p.add_argument("--data-dir", type=Path, default=Path("./data"),
                   help="Root data directory (default: ./data)")
    p.add_argument("--tpch-dir", type=Path, default=None,
                   help="Dir containing TPC-H *.parquet files "
                        "(default: <data-dir>/tpch-sf1/parquet)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir

    vech_parquet = data_dir / "vech" / "parquet"
    vech_csv     = data_dir / "vech"
    tpch         = args.tpch_dir or data_dir / "tpch-sf1" / "parquet"

    print(f"\n{BOLD}Validating dataset under: {data_dir.resolve()}{RESET}")
    print(f"{BOLD}TPC-H dir: {tpch}{RESET}")

    check_files(vech_parquet, vech_csv, tpch)

    tables = load_tables(vech_parquet, vech_csv, tpch)
    if not tables:
        fail("No tables loaded — cannot continue")
        print_summary()
        sys.exit(1)

    counts = check_row_counts(tables)
    report_scale_factor(tables)
    check_csv_parity(tables, counts)
    check_schema(tables)
    check_nulls(tables)
    check_duplicates(tables)
    check_foreign_keys(tables)
    check_embeddings(tables)
    check_value_ranges(tables)
    print_summary()

    if _failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
