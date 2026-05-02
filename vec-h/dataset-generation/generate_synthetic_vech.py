#!/usr/bin/env python3
"""Generate synthetic VECH datasets for performance testing.

Creates a complete VECH-compatible dataset directory at any scale factor,
with random vectors and realistic distributions fitted to actual SF1 Ind&Sci data.
Output is directly loadable by Maximus maxbench (--path <output_dir> --benchmark vech).

Usage:
    # SF1 test (fast, ~1 min)
    uv run python3 scripts/generate_synthetic_vech.py --sf 1

    # SF30 perf test (~15 min, ~34 GB RAM for TPC-H + streaming writes)
    uv run python3 scripts/generate_synthetic_vech.py --sf 30

    # Custom dimensions, skip TPC-H if already generated
    uv run python3 scripts/generate_synthetic_vech.py --sf 30 --rev_dim 256 --skip_tpch
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# =============================================================================
# Distribution parameters fitted to actual SF1 Industrial & Scientific data
# =============================================================================

# Reviews per part: heavy long tail (mean=12, median=2, skew=72.5)
REVIEW_MU = 0.693      # ln(median) = ln(2)
REVIEW_SIGMA = 1.89    # fitted from mean/median ratio = 6
MEAN_REV_PER_PART = 12

# Images per part: concentrated, near-normal (mean=3.76, median=3, skew=0.62)
IMAGE_MU = 1.099       # ln(median) = ln(3)
IMAGE_SIGMA = 0.67     # fitted from mean/median ratio = 1.25
MEAN_IMG_PER_PART = 3.76

# Rating distribution from actual SF1 data (for pre/post-filter queries)
RATING_VALUES = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
RATING_PROBS = np.array([0.075, 0.038, 0.063, 0.121, 0.703])

# Image variant distribution from actual SF1 data (~25% MAIN)
VARIANT_VALUES = ["MAIN", "PT01", "PT02", "PT03", "PT04", "PT05", "PT06", "PT07", "PT08"]
VARIANT_PROBS = np.array([0.248, 0.170, 0.143, 0.122, 0.104, 0.088, 0.063, 0.029, 0.017])
VARIANT_PROBS = VARIANT_PROBS / VARIANT_PROBS.sum()  # normalize

# TPC-H constants
PARTS_PER_SF = 200_000
CUSTOMERS_PER_SF = 150_000

# Parquet schemas (match actual SF1 pipeline output exactly)
REVIEWS_SCHEMA = pa.schema([
    ("rv_rating", pa.float32()),
    ("rv_helpful_vote", pa.int32()),
    ("rv_title", pa.string()),
    ("rv_text", pa.string()),
    ("rv_embedding", pa.large_list(pa.float32())),
    ("rv_partkey", pa.int64()),
    ("rv_custkey", pa.int64()),
    ("rv_reviewkey", pa.int64()),
])

REVIEWS_QUERIES_SCHEMA = pa.schema([
    ("rv_rating_queries", pa.float32()),
    ("rv_helpful_vote_queries", pa.int32()),
    ("rv_title_queries", pa.string()),
    ("rv_text_queries", pa.string()),
    ("rv_embedding_queries", pa.large_list(pa.float32())),
    ("rv_partkey_queries", pa.int64()),
    ("rv_custkey_queries", pa.int64()),
    ("rv_reviewkey_queries", pa.int64()),
])

IMAGES_SCHEMA = pa.schema([
    ("i_image_url", pa.string()),
    ("i_variant", pa.string()),
    ("i_embedding", pa.large_list(pa.float32())),
    ("i_imagekey", pa.int64()),
    ("i_partkey", pa.int64()),
])

IMAGES_QUERIES_SCHEMA = pa.schema([
    ("i_image_url_queries", pa.string()),
    ("i_variant_queries", pa.string()),
    ("i_embedding_queries", pa.large_list(pa.float32())),
    ("i_imagekey_queries", pa.int64()),
    ("i_partkey_queries", pa.int64()),
])


# =============================================================================
# Distribution generation
# =============================================================================

def generate_counts(n_total, n_parts, mu, sigma, rng):
    """Generate per-part item counts following a log-normal distribution.

    Returns an int64 array of length n_parts that sums to exactly n_total,
    with each element >= 1.
    """
    counts = np.maximum(1, np.round(rng.lognormal(mu, sigma, n_parts))).astype(np.int64)
    # Scale to hit exact total
    counts = np.maximum(1, np.round(counts * (n_total / counts.sum()))).astype(np.int64)

    # Fix rounding error to guarantee exact sum
    diff = int(n_total - counts.sum())
    while diff != 0:
        if diff > 0:
            indices = rng.choice(n_parts, size=min(abs(diff), n_parts), replace=False)
            for idx in indices[:abs(diff)]:
                counts[idx] += 1
                diff -= 1
        else:
            candidates = np.where(counts > 1)[0]
            indices = rng.choice(candidates, size=min(abs(diff), len(candidates)), replace=False)
            for idx in indices[:abs(diff)]:
                counts[idx] -= 1
                diff += 1

    return counts


def counts_to_partkeys(counts, cumsum, start, end):
    """Map global row indices [start, end) to 1-indexed partkeys using cumsum."""
    row_indices = np.arange(start, end)
    partkeys = np.searchsorted(cumsum, row_indices, side="right").astype(np.int64) + 1
    return partkeys


def make_embedding_column(vectors):
    """Convert (N, dim) float32 numpy array to a pa.large_list(float32) column."""
    n = vectors.shape[0]
    flat = vectors.ravel()
    dim = vectors.shape[1]
    offsets = np.arange(0, (n + 1) * dim, dim, dtype=np.int64)
    return pa.LargeListArray.from_arrays(pa.array(offsets), pa.array(flat))


# =============================================================================
# Shard writers
# =============================================================================

def write_review_shards(output_dir, n_reviews, n_customers, rev_dim,
                        rev_cumsum, shard_size, rng):
    """Stream-write review parquet shards."""
    n_shards = math.ceil(n_reviews / shard_size)
    empty_strings = None  # lazily allocated

    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, n_reviews)
        batch = end - start

        # Partkeys from distribution
        partkeys = counts_to_partkeys(rev_cumsum, rev_cumsum, start, end)

        # Random vectors
        shard_rng = np.random.default_rng(rng.integers(0, 2**63))
        vectors = shard_rng.standard_normal((batch, rev_dim)).astype(np.float32)

        # Metadata
        ratings = shard_rng.choice(RATING_VALUES, size=batch, p=RATING_PROBS)
        helpful = (shard_rng.geometric(p=0.3, size=batch) - 1).astype(np.int32)

        if empty_strings is None or len(empty_strings) != batch:
            empty_strings = pa.array([""] * batch, type=pa.string())

        table = pa.table({
            "rv_rating": pa.array(ratings),
            "rv_helpful_vote": pa.array(helpful),
            "rv_title": empty_strings,
            "rv_text": empty_strings,
            "rv_embedding": make_embedding_column(vectors),
            "rv_partkey": pa.array(partkeys),
            "rv_custkey": pa.array(
                shard_rng.integers(1, n_customers + 1, size=batch, dtype=np.int64)
            ),
            "rv_reviewkey": pa.array(
                np.arange(start, end, dtype=np.int64)
            ),
        }, schema=REVIEWS_SCHEMA)

        path = output_dir / f"reviews-{shard_idx}.parquet"
        pq.write_table(table, path)

        del vectors, table
        print(f"  reviews-{shard_idx}.parquet  ({batch:>10,} rows)", flush=True)


def write_review_queries(output_dir, n_queries, rev_dim, rev_counts,
                         n_customers, rng):
    """Write reviews_queries.parquet."""
    # Sample partkeys from parts that actually have reviews
    parts_with_reviews = np.where(rev_counts > 0)[0] + 1  # 1-indexed
    query_partkeys = rng.choice(parts_with_reviews, size=n_queries).astype(np.int64)

    vectors = rng.standard_normal((n_queries, rev_dim)).astype(np.float32)
    ratings = rng.choice(RATING_VALUES, size=n_queries, p=RATING_PROBS)
    helpful = (rng.geometric(p=0.3, size=n_queries) - 1).astype(np.int32)
    empty = pa.array([""] * n_queries, type=pa.string())

    table = pa.table({
        "rv_rating_queries": pa.array(ratings),
        "rv_helpful_vote_queries": pa.array(helpful),
        "rv_title_queries": empty,
        "rv_text_queries": empty,
        "rv_embedding_queries": make_embedding_column(vectors),
        "rv_partkey_queries": pa.array(query_partkeys),
        "rv_custkey_queries": pa.array(
            rng.integers(1, n_customers + 1, size=n_queries, dtype=np.int64)
        ),
        "rv_reviewkey_queries": pa.array(
            np.arange(n_queries, dtype=np.int64)
        ),
    }, schema=REVIEWS_QUERIES_SCHEMA)

    path = output_dir / "reviews_queries.parquet"
    pq.write_table(table, path)
    print(f"  reviews_queries.parquet     ({n_queries:>10,} rows)", flush=True)


def write_image_shards(output_dir, n_images, img_dim, img_cumsum,
                       shard_size, rng):
    """Stream-write image parquet shards."""
    n_shards = math.ceil(n_images / shard_size)

    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, n_images)
        batch = end - start

        partkeys = counts_to_partkeys(img_cumsum, img_cumsum, start, end)

        shard_rng = np.random.default_rng(rng.integers(0, 2**63))
        vectors = shard_rng.standard_normal((batch, img_dim)).astype(np.float32)

        # Variant distribution
        variants = shard_rng.choice(VARIANT_VALUES, size=batch, p=VARIANT_PROBS)

        empty = pa.array([""] * batch, type=pa.string())

        table = pa.table({
            "i_image_url": empty,
            "i_variant": pa.array(variants, type=pa.string()),
            "i_embedding": make_embedding_column(vectors),
            "i_imagekey": pa.array(np.arange(start, end, dtype=np.int64)),
            "i_partkey": pa.array(partkeys),
        }, schema=IMAGES_SCHEMA)

        path = output_dir / f"images-{shard_idx}.parquet"
        pq.write_table(table, path)

        del vectors, table
        print(f"  images-{shard_idx}.parquet   ({batch:>10,} rows)", flush=True)


def write_image_queries(output_dir, n_queries, img_dim, img_counts, rng):
    """Write images_queries.parquet."""
    parts_with_images = np.where(img_counts > 0)[0] + 1
    query_partkeys = rng.choice(parts_with_images, size=n_queries).astype(np.int64)

    vectors = rng.standard_normal((n_queries, img_dim)).astype(np.float32)
    variants = rng.choice(VARIANT_VALUES, size=n_queries, p=VARIANT_PROBS)
    empty = pa.array([""] * n_queries, type=pa.string())

    table = pa.table({
        "i_image_url_queries": empty,
        "i_variant_queries": pa.array(variants, type=pa.string()),
        "i_embedding_queries": make_embedding_column(vectors),
        "i_imagekey_queries": pa.array(np.arange(n_queries, dtype=np.int64)),
        "i_partkey_queries": pa.array(query_partkeys),
    }, schema=IMAGES_QUERIES_SCHEMA)

    path = output_dir / "images_queries.parquet"
    pq.write_table(table, path)
    print(f"  images_queries.parquet      ({n_queries:>10,} rows)", flush=True)


# =============================================================================
# TPC-H generation
# =============================================================================

def generate_tpch(sf, output_dir):
    """Generate TPC-H tables via DuckDB COPY TO (streaming, low memory)."""
    import duckdb

    print(f"\nGenerating TPC-H SF{sf}...")
    t0 = time.time()

    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={sf})")

    tables = ["customer", "lineitem", "nation", "orders",
              "part", "partsupp", "region", "supplier"]
    for t in tables:
        path = output_dir / f"{t}.parquet"
        con.execute(f"COPY {t} TO '{path}' (FORMAT PARQUET)")
        print(f"  {t}.parquet", flush=True)

    con.close()
    print(f"  TPC-H done in {time.time() - t0:.1f}s")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic VECH dataset for performance testing."
    )
    parser.add_argument("--sf", type=float, default=1,
                        help="TPC-H scale factor (default: 1)")
    parser.add_argument("--rev_dim", type=int, default=128,
                        help="Review embedding dimension (default: 128)")
    parser.add_argument("--img_dim", type=int, default=128,
                        help="Image embedding dimension (default: 128)")
    parser.add_argument("--n_queries", type=int, default=10_000,
                        help="Number of query vectors (default: 10000)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: data-synthetic-sf_{sf})")
    parser.add_argument("--shard_size", type=int, default=1_000_000,
                        help="Rows per parquet shard (default: 1000000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--skip_tpch", action="store_true",
                        help="Skip TPC-H table generation")
    args = parser.parse_args()

    sf = args.sf
    sf_label = int(sf) if sf == int(sf) else sf

    n_parts = int(PARTS_PER_SF * sf)
    n_customers = int(CUSTOMERS_PER_SF * sf)
    n_reviews = int(n_parts * MEAN_REV_PER_PART)
    n_images = int(n_parts * MEAN_IMG_PER_PART)

    output_dir = Path(args.output_dir) if args.output_dir else Path(f"data-synthetic-sf_{sf_label}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    print(f"=== Synthetic VECH Generator ===")
    print(f"  SF:          {sf}")
    print(f"  Parts:       {n_parts:,}")
    print(f"  Customers:   {n_customers:,}")
    print(f"  Reviews:     {n_reviews:,} ({args.rev_dim}d)")
    print(f"  Images:      {n_images:,} ({args.img_dim}d)")
    print(f"  Queries:     {args.n_queries:,}")
    print(f"  Shard size:  {args.shard_size:,}")
    print(f"  Output:      {output_dir}")
    print()

    # --- Generate distributions ---
    print("Generating review distribution (log-normal, sigma=1.89)...")
    t0 = time.time()
    rev_counts = generate_counts(n_reviews, n_parts, REVIEW_MU, REVIEW_SIGMA, rng)
    rev_cumsum = np.cumsum(rev_counts)
    print(f"  mean={rev_counts.mean():.1f}, median={np.median(rev_counts):.0f}, "
          f"p99={np.percentile(rev_counts, 99):.0f}, max={rev_counts.max()}")

    print("Generating image distribution (log-normal, sigma=0.67)...")
    img_counts = generate_counts(n_images, n_parts, IMAGE_MU, IMAGE_SIGMA, rng)
    img_cumsum = np.cumsum(img_counts)
    print(f"  mean={img_counts.mean():.1f}, median={np.median(img_counts):.0f}, "
          f"p99={np.percentile(img_counts, 99):.0f}, max={img_counts.max()}")
    print(f"  Distributions ready in {time.time() - t0:.1f}s")
    print()

    # --- Write reviews ---
    print(f"Writing reviews ({math.ceil(n_reviews / args.shard_size)} shards)...")
    t0 = time.time()
    write_review_shards(output_dir, n_reviews, n_customers, args.rev_dim,
                        rev_cumsum, args.shard_size, rng)
    write_review_queries(output_dir, args.n_queries, args.rev_dim,
                         rev_counts, n_customers, rng)
    print(f"  Reviews done in {time.time() - t0:.1f}s")
    print()

    # --- Write images ---
    print(f"Writing images ({math.ceil(n_images / args.shard_size)} shards)...")
    t0 = time.time()
    write_image_shards(output_dir, n_images, args.img_dim, img_cumsum,
                       args.shard_size, rng)
    write_image_queries(output_dir, args.n_queries, args.img_dim,
                        img_counts, rng)
    print(f"  Images done in {time.time() - t0:.1f}s")
    print()

    # --- TPC-H ---
    if not args.skip_tpch:
        generate_tpch(sf, output_dir)
    else:
        print("Skipping TPC-H generation (--skip_tpch)")
    print()

    # --- Summary ---
    total_bytes = sum(f.stat().st_size for f in output_dir.glob("*.parquet"))
    print(f"=== Done ===")
    print(f"  Output:      {output_dir}")
    print(f"  Total size:  {total_bytes / (1024**3):.2f} GB")
    print(f"  Files:       {len(list(output_dir.glob('*.parquet')))}")


if __name__ == "__main__":
    main()
