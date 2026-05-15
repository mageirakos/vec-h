import logging
import argparse
import math
from pathlib import Path
from typing import Optional, List, Union

import polars as pl


logger = logging.getLogger(__name__)

REVIEWS_SCHEMA = [
    ("rv_reviewkey",    pl.Int64),
    ("rv_rating",       pl.Float32),
    ("rv_helpful_vote", pl.Int32),
    ("rv_title",        pl.Utf8),
    ("rv_text",         pl.Utf8),
    ("rv_embedding",    None),
    ("rv_partkey",      pl.Int64),
    ("rv_custkey",      pl.Int64),
]

IMAGES_SCHEMA = [
    ("i_imagekey",  pl.Int64),
    ("i_image_url", pl.Utf8),
    ("i_variant",   pl.Utf8),
    ("i_embedding", None),
    ("i_partkey",   pl.Int64),
]

_EMBEDDING_COLS = {"rv_embedding", "i_embedding", "rv_embedding_queries", "i_embedding_queries"}


def _enforce_schema(lf: pl.LazyFrame, schema, queries: bool = False) -> pl.LazyFrame:
    """Project to the declared column order and cast each column to its dtype.

    When queries=True, every column name gets the `_queries` suffix.
    Columns with dtype=None are projected without a cast.
    """
    suffix = "_queries" if queries else ""
    exprs = []
    for name, dtype in schema:
        col = pl.col(name + suffix)
        if dtype is not None:
            col = col.cast(dtype)
        exprs.append(col)
    return lf.select(exprs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Amazon Reviews to TPC-H Mapping Pipeline"
    )
    parser.add_argument(
        "--input_dir_imgs_parquet",
        type=Path,
        help="Directory containing Amazon image datasets"
    )
    parser.add_argument(
        "--input_dir_reviews_parquet",
        type=Path,
        help="Directory containing Amazon reviews datasets"
    )
    parser.add_argument(
        "--input_dir_tpch_parquet",
        type=Path,
        help="Directory containing TPC-H parquet files, e.g. customer*.parquet, part*.parquet, etc."
    )
    parser.add_argument(
        "--n_query_samples",
        type=int,
        default=10000,
        help="Number of samples for query datasets",
    )
    parser.add_argument(
        "--n_output_files",
        type=int,
        default=1,
        help="Split data tables into this many files (e.g. reviews-0.parquet). Query tables are always a single file."
    )
    parser.add_argument(
        "--output_dir_parquet",
        type=Path,
        default=None,
        help="Where to write parquet output files",
    )
    parser.add_argument(
        "--output_dir_csv",
        type=Path,
        default=None,
        help="Where to write CSV output (embedding columns are dropped)",
    )
    parser.add_argument(
        "--output_dir_tbl",
        type=Path,
        default=None,
        help="Where to write TPC-H .tbl output (pipe-delimited, embedding columns are dropped)",
    )
    parser.add_argument(
        "--seed", type=int, default=98, help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    if args.output_dir_parquet is None and args.output_dir_csv is None and args.output_dir_tbl is None:
        parser.error("At least one of --output_dir_parquet, --output_dir_csv, or --output_dir_tbl must be provided.")
    return args


def _load_parquet_dataset(
    path: Union[str, Path],
    columns: Optional[List[str]] = None,
    nrows: Optional[int] = None,
) -> pl.LazyFrame:
    """Load all parquet files matching path (directory or prefix) into a single DataFrame.
    """
    p = str(path)
    files = sorted(
        str(f) for f in path.parent.rglob("*.parquet")
        if str(f).startswith(p + "/") or str(f).startswith(p + ".") or str(f).startswith(p + "_")
    )
    if not files:
        raise FileNotFoundError(f"No parquet files found for path: {path}")

    logger.info(f"At {path} loading files: {files}")

    # Validate schema consistency across files (metadata only)
    file_schemas = {f: set(pl.read_parquet_schema(f).names()) for f in files}
    if columns is None:
        all_cols = list(file_schemas.values())
        if any(s != all_cols[0] for s in all_cols[1:]):
            mismatches = {f: s for f, s in file_schemas.items() if s != all_cols[0]}
            raise ValueError(f"Column mismatch across parquet files: {mismatches}")
    else:
        required = set(columns)
        missing = {f: required - s for f, s in file_schemas.items() if required - s}
        if missing:
            raise ValueError(f"Files missing required columns: {missing}")

    lf = pl.scan_parquet(files, n_rows=nrows).select(columns or pl.all())
    return lf

def create_part_mapping(
    lf_reviews: pl.LazyFrame, lf_images: pl.LazyFrame, lf_part: pl.LazyFrame, seed: int
) -> pl.DataFrame:
    """Map Amazon parent_asins to TPC-H p_partkeys.

    Collects unique ASINs from both reviews and images so the mapping is
    consistent across both tables. Randomly pairs min(n_asins, n_partkeys)
    from each side — no IDs reused.

    Returns a DataFrame with columns [parent_asin, p_partkey].
    """
    all_asins = pl.concat([
        lf_reviews.select("parent_asin").collect()["parent_asin"],
        lf_images.select("parent_asin").collect()["parent_asin"],
    ]).unique().sort()
    partkeys = lf_part.select("p_partkey").collect()["p_partkey"].unique().sort()
    n_asins, n_parts = len(all_asins), len(partkeys)
    limit = min(n_asins, n_parts)

    logger.info(f"[Part Mapping] {n_asins:,} ASINs x {n_parts:,} partkeys -> {limit:,} pairs")

    return pl.DataFrame({
        "parent_asin": all_asins.sample(n=limit, seed=seed, shuffle=True),
        "p_partkey": partkeys.sample(n=limit, seed=seed, shuffle=True),
    })


def process_reviews(
    lf: pl.LazyFrame,
    part_mapping: pl.DataFrame,
    lf_cust: pl.LazyFrame,
    seed: int,
) -> pl.LazyFrame:
    # Map to columns and deduplicate
    lf = lf.drop(["asin", "timestamp", "verified_purchase"], strict=False).rename({
        "rating": "rv_rating",
        "title": "rv_title",
        "text": "rv_text",
        "embedding": "rv_embedding",
        "helpful_vote": "rv_helpful_vote",
    })
    dedup_cols = [c for c in lf.collect_schema().names() if c not in {"rv_embedding", "_row_idx"}]
    n_before = lf.select(pl.len()).collect().item()
    lf = lf.unique(subset=dedup_cols, keep="first")
    n_after = lf.select(pl.len()).collect().item()
    logger.info(f"[Reviews] dedup: {n_before:,} -> {n_after:,} rows ({n_before - n_after:,} removed)")

    # Inner join drops unmapped parent_asins in one step (no left + drop_nulls needed)
    lf = (
        lf.join(part_mapping.lazy(), on="parent_asin", how="inner")
        .rename({"p_partkey": "rv_partkey"})
        .drop("parent_asin")
    )

    user_ids = lf.select("user_id").collect()["user_id"].unique().sort()
    custkeys = lf_cust.select("c_custkey").collect()["c_custkey"].unique().sort()
    n_users, n_cust = len(user_ids), len(custkeys)
    repeat = math.ceil(n_users / n_cust)
    cust_mapping = pl.DataFrame({
        "user_id": user_ids.shuffle(seed=seed),
        "rv_custkey": pl.concat([custkeys] * repeat).head(n_users).sample(n=n_users, seed=seed, shuffle=True),
    })
    logger.info(f"[Customer Mapping] {n_users:,} users x {n_cust:,} custkeys -> {n_users:,} pairs")

    n_before = lf.select(pl.len()).collect().item()
    lf = (
        lf.join(cust_mapping.lazy(), on="user_id", how="inner")
        .drop("user_id")
        .with_row_index("rv_reviewkey")
        .with_columns(pl.col("rv_reviewkey").cast(pl.Int64()))
    )
    n_after_cust = lf.select(pl.len()).collect().item()
    if n_after_cust != n_before:
        raise ValueError(f"Customer mapping join dropped {n_before - n_after_cust} rows — some user_ids unmatched")

    # Lock dtypes downstream consumers depend on (Maximus expects int64 keys,
    # int32 helpful_vote, float32 rating; postgres loader looks up by name).
    lf = lf.with_columns([
        pl.col("rv_partkey").cast(pl.Int64),
        pl.col("rv_custkey").cast(pl.Int64),
        pl.col("rv_helpful_vote").cast(pl.Int32),
        pl.col("rv_rating").cast(pl.Float32),
    ])
    return lf

def process_images(lf: pl.LazyFrame, part_mapping: pl.DataFrame) -> pl.LazyFrame:
    lf = (lf.drop(["image_index"], strict=False)
            .rename({"image_url": "i_image_url", "variant": "i_variant", "embedding": "i_embedding"}))

    bad_gif = "https://m.media-amazon.com/images/I/01RmK+J4pJL.gif"
    dedup_cols = [c for c in lf.collect_schema().names() if c not in ("i_embedding", "_row_idx")]
    n_before = lf.select(pl.len()).collect().item()
    lf = lf.unique(subset=dedup_cols, keep="first").filter(pl.col("i_image_url") != bad_gif)
    n_after = lf.select(pl.len()).collect().item()
    logger.info(f"[Images] dedup+filter: {n_before:,} -> {n_after:,} rows ({n_before - n_after:,} removed)")
    lf = (lf.join(part_mapping.lazy(), on="parent_asin", how="inner")
            .rename({"p_partkey": "i_partkey"}).drop("parent_asin")
            .with_row_index("i_imagekey"))

    lf = lf.with_columns([
        pl.col("i_partkey").cast(pl.Int64),
        pl.col("i_imagekey").cast(pl.Int64),
    ])
    return lf


def split_query(
    lf: pl.LazyFrame, key_col: str, n: int, seed: int
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Split a LazyFrame into (data, query) sets by sampling _row_idx.

    Drops _row_idx and reassigns key_col sequentially in both halves.
    Query columns are renamed with a _queries suffix.
    Returns (lf_data, lf_query).
    """
    all_idx = lf.select("_row_idx").collect()["_row_idx"]
    n_total = len(all_idx)
    query_idx = all_idx.sample(n=n, seed=seed)
    rename_cols = [c for c in lf.collect_schema().names() if c not in ("_row_idx", key_col)]
    query_idx_list = query_idx.to_list()

    lf_query = (lf.filter(pl.col("_row_idx").is_in(query_idx_list))
                  .sort("_row_idx")
                  .drop(["_row_idx", key_col])
                  .rename({c: c + "_queries" for c in rename_cols})
                  .with_row_index(key_col + "_queries")
                  .with_columns(pl.col(key_col + "_queries").cast(pl.Int64)))
    lf_data = (lf.filter(~pl.col("_row_idx").is_in(query_idx_list))
                 .sort("_row_idx")
                 .drop(["_row_idx", key_col])
                 .with_row_index(key_col)
                 .with_columns(pl.col(key_col).cast(pl.Int64)))
    n_data = lf_data.select(pl.len()).collect().item()
    n_query = lf_query.select(pl.len()).collect().item()
    if n_data + n_query != n_total:
        raise ValueError(f"Split lost rows: {n_data} + {n_query} != {n_total}")
    return lf_data, lf_query


def _split(lf: pl.LazyFrame, n_files: int):
    """Yield (slice_lf, suffix) pairs for writing a table across n_files files.
    If n_files == 1, yields the full frame with an empty suffix.

    Suffix format is ``-{i}`` so the resulting filenames (e.g. ``reviews-0.parquet``)
    match Maximus's catalogue regex ``^<table>(-\\d+)?\\.(csv|parquet)$``
    (src/maximus/database_catalogue.cpp).
    """
    if n_files == 1:
        yield lf, ""
        return
    n_rows = lf.select(pl.len()).collect().item()
    batch_size = math.ceil(n_rows / n_files)
    for i in range(n_files):
        yield lf.slice(i * batch_size, batch_size), f"-{i}"


def write_outputs(
    lf_reviews: pl.LazyFrame,
    lf_reviews_query: pl.LazyFrame,
    lf_images: pl.LazyFrame,
    lf_images_query: pl.LazyFrame,
    config,
) -> None:
    n = config.n_output_files

    lf_reviews       = _enforce_schema(lf_reviews,       REVIEWS_SCHEMA, queries=False)
    lf_reviews_query = _enforce_schema(lf_reviews_query, REVIEWS_SCHEMA, queries=True)
    lf_images        = _enforce_schema(lf_images,        IMAGES_SCHEMA,  queries=False)
    lf_images_query  = _enforce_schema(lf_images_query,  IMAGES_SCHEMA,  queries=True)

    def _drop_embeddings(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.drop([c for c in lf.collect_schema().names() if c in _EMBEDDING_COLS])

    def write_one(lf, path, **kwargs):
        logger.info(f"  Writing {path}")
        if str(path).endswith(".parquet"):
            lf.sink_parquet(path, compression="snappy")
        else:
            lf.sink_csv(path, **kwargs)

    # Write one table at a time so only one embedding column set is in flight.
    # reviews (with embeddings) is ~20 GB; doing all sinks in parallel via
    # pl.collect_all() caused OOM on DGX.
    if config.output_dir_parquet:
        config.output_dir_parquet.mkdir(parents=True, exist_ok=True)
        d = config.output_dir_parquet
        for lf_slice, sfx in _split(lf_reviews, n):
            write_one(lf_slice, d / f"reviews{sfx}.parquet")
        write_one(lf_reviews_query, d / "reviews_queries.parquet")
        for lf_slice, sfx in _split(lf_images, n):
            write_one(lf_slice, d / f"images{sfx}.parquet")
        write_one(lf_images_query, d / "images_queries.parquet")

    if config.output_dir_csv:
        config.output_dir_csv.mkdir(parents=True, exist_ok=True)
        d = config.output_dir_csv
        for lf_slice, sfx in _split(_drop_embeddings(lf_reviews), n):
            write_one(lf_slice, d / f"reviews{sfx}.csv")
        write_one(_drop_embeddings(lf_reviews_query), d / "reviews_queries.csv")
        for lf_slice, sfx in _split(_drop_embeddings(lf_images), n):
            write_one(lf_slice, d / f"images{sfx}.csv")
        write_one(_drop_embeddings(lf_images_query), d / "images_queries.csv")

    if config.output_dir_tbl:
        config.output_dir_tbl.mkdir(parents=True, exist_ok=True)
        d = config.output_dir_tbl
        for lf_slice, sfx in _split(_drop_embeddings(lf_reviews), n):
            write_one(lf_slice, d / f"reviews{sfx}.tbl", separator="|")
        write_one(_drop_embeddings(lf_reviews_query), d / "reviews_queries.tbl", separator="|")
        for lf_slice, sfx in _split(_drop_embeddings(lf_images), n):
            write_one(lf_slice, d / f"images{sfx}.tbl", separator="|")
        write_one(_drop_embeddings(lf_images_query), d / "images_queries.tbl", separator="|")


def run_pipeline(config):
    """Build a lazy transformation plan over reviews and images, then stream to output sinks.

    We exploit polars' lazy evaluation to avoid ever materializing embedding columns in memory.
    Embeddings are large (e.g. 1536 floats per row) and would dwarf all other data combined.
    All intermediate collect() calls are narrow (IDs, counts only) — polars projection pushdown
    ensures embedding columns are skipped. Only at write_outputs() do embeddings touch memory,
    and even then only transiently: sink_parquet(lazy=True) + pl.collect_all() processes data
    in fixed-size chunks via polars' streaming engine.
    """
    logger.info("Lazily loading datasets...")
    lf_part = _load_parquet_dataset(config.input_dir_tpch_parquet / "part")
    lf_customer = _load_parquet_dataset(config.input_dir_tpch_parquet / "customer")
    lf_reviews = _load_parquet_dataset(config.input_dir_reviews_parquet)
    lf_images = _load_parquet_dataset(config.input_dir_imgs_parquet)

    logger.info("Creating new tables...")
    lf_reviews = lf_reviews.with_row_index("_row_idx")
    lf_images = lf_images.with_row_index("_row_idx")
    part_mapping = create_part_mapping(lf_reviews, lf_images, lf_part, config.seed)
    lf_reviews = process_reviews(lf_reviews, part_mapping, lf_customer, config.seed)
    lf_images = process_images(lf_images, part_mapping)

    logger.info("Splitting tables into data vs query sets...")
    n_reviews = lf_reviews.select(pl.len()).collect().item()
    n_images = lf_images.select(pl.len()).collect().item()
    n_query_maximum = min(n_reviews, n_images)
    if config.n_query_samples > n_query_maximum:
        raise ValueError(f"Insufficient data for requested number of query samples (<{n_query_maximum:,})")
    lf_reviews, lf_reviews_query = split_query(lf_reviews, "rv_reviewkey", config.n_query_samples, config.seed)
    lf_images, lf_images_query = split_query(lf_images, "i_imagekey", config.n_query_samples, config.seed)

    logger.info("Writing outputs...")
    write_outputs(lf_reviews, lf_reviews_query, lf_images, lf_images_query, config)

    logger.info("Output Sizes (Rows):")
    logger.info(f"  - Mapping (ASIN <-> partkey): {len(part_mapping):,} unique products matched")
    logger.info(f"  - Reviews (Data):       {n_reviews - config.n_query_samples:,}")
    logger.info(f"  - Reviews (Queries):    {config.n_query_samples:,}")
    logger.info(f"  - Images (Data):        {n_images - config.n_query_samples:,}")
    logger.info(f"  - Images (Queries):     {config.n_query_samples:,}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    config = parse_args()
    logger.info("Running table generation with args: %s", config)
    run_pipeline(config)
