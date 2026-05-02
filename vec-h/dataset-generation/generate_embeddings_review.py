#!/usr/bin/env python3
"""
Generate review text embeddings for Amazon product reviews using Qwen3-Embedding (default) or Jina v4.

Reads all *.jsonl.gz files from --input-dir and writes partitioned parquet to --output-dir.

Usage:
    python generate_embeddings_review.py --input-dir data/amazon-23/raw/reviews --output-dir data/amazon-23/parquet/Qwen0.6B/reviews
    python generate_embeddings_review.py --input-dir data/amazon-23/raw/reviews --output-dir data/amazon-23/parquet/Qwen4B/reviews --model-size 4B --batch-size 16

Long-running:
    nohup python generate_embeddings_review.py ... >> embedding_review.log 2>&1 &
"""

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Generator, List, Optional

import numpy as np
import pyarrow as pa
from tqdm import tqdm

from generate_embeddings_img import (
    make_generator,
    PartitionedParquetWriter,
)

# logging is configured by generate_embeddings_img at import time
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate review text embeddings for Amazon product reviews.")
    parser.add_argument("--input-dir", required=True, help="Directory containing review *.jsonl.gz files")
    parser.add_argument("--output-dir", required=True, help="Directory to write parquet files")
    parser.add_argument("--model-name", default="qwen",
                        help="Model to use: 'qwen' (default) or 'jina-v4'")
    parser.add_argument("--model-size", default="0.6B", choices=["0.6B", "4B", "8B"],
                        help="Qwen model size (default: 0.6B, ignored for jina-v4)")
    parser.add_argument("--max-seq-length", type=int, default=None,
                        help="Max sequence length in tokens for Qwen (default: model's native limit, ~32768)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 precision")
    parser.add_argument("--partition-size-gb", type=float, default=1.0,
                        help="Target size per partition file in GB (default: 1.0)")
    parser.add_argument("--rows-per-rowgroup", type=int, default=100000,
                        help="Rows per parquet row group (default: 100000)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def stream_reviews_from_jsonl_gz(filepath: Path) -> Generator[Dict, None, None]:
    """Stream reviews from a .jsonl.gz file."""
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                yield json.loads(line.strip())
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse line %d in %s: %s", idx, filepath.name, e)


# ---------------------------------------------------------------------------
# Parquet schema and table creation
# ---------------------------------------------------------------------------

def get_schema() -> pa.Schema:
    return pa.schema([
        pa.field('asin',              pa.string()),
        pa.field('parent_asin',       pa.string()),
        pa.field('user_id',           pa.string()),
        pa.field('timestamp',         pa.int64()),
        pa.field('rating',            pa.float32()),
        pa.field('helpful_vote',      pa.int32()),
        pa.field('verified_purchase', pa.bool_()),
        pa.field('title',             pa.string()),
        pa.field('text',              pa.string()),
        pa.field('embedding',         pa.large_list(pa.float32())),
    ])


def create_pyarrow_table(metadata_list: List[Dict], embeddings_np: np.ndarray, embedding_dim: int) -> pa.Table:
    n_rows = len(metadata_list)
    flat_embeddings = embeddings_np.flatten().astype(np.float32)
    offsets = np.arange(0, (n_rows + 1) * embedding_dim, embedding_dim, dtype=np.int64)
    embedding_array = pa.LargeListArray.from_arrays(
        pa.array(offsets, type=pa.int64()),
        pa.array(flat_embeddings, type=pa.float32()),
    )
    return pa.table({
        'asin':              pa.array([m.get('asin')                     for m in metadata_list], type=pa.string()),
        'parent_asin':       pa.array([m.get('parent_asin')              for m in metadata_list], type=pa.string()),
        'user_id':           pa.array([m.get('user_id')                  for m in metadata_list], type=pa.string()),
        'timestamp':         pa.array([m.get('timestamp')                for m in metadata_list], type=pa.int64()),
        'rating':            pa.array([m.get('rating')                   for m in metadata_list], type=pa.float32()),
        'helpful_vote':      pa.array([m.get('helpful_vote', 0)          for m in metadata_list], type=pa.int32()),
        'verified_purchase': pa.array([m.get('verified_purchase', False) for m in metadata_list], type=pa.bool_()),
        'title':             pa.array([m.get('title', '')                for m in metadata_list], type=pa.string()),
        'text':              pa.array([m.get('text', '')                 for m in metadata_list], type=pa.string()),
        'embedding':         embedding_array,
    })


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_reviews(
    input_dir: Path,
    output_dir: Path,
    model_name: str = "qwen",
    model_size: str = "0.6B",
    max_seq_length: Optional[int] = None,
    batch_size: int = 8,
    use_fp16: bool = True,
    partition_size_gb: float = 1.0,
    rows_per_rowgroup: int = 100000,
):
    input_files = sorted(input_dir.glob("*.jsonl.gz"))
    if not input_files:
        logger.error("No *.jsonl.gz files found in %s", input_dir)
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    name_lower = model_name.lower()
    model_label = f"{model_name} ({model_size})" if name_lower == "qwen" else model_name
    logger.info("Model: %s | Batch: %d | Files: %d", model_label, batch_size, len(input_files))
    for f in input_files:
        logger.info("  %s", f.name)

    generator = make_generator(model_name, batch_size=batch_size, use_fp16=use_fp16,
                               model_size=model_size, max_seq_length=max_seq_length)
    writer = PartitionedParquetWriter(
        base_path=output_dir, schema=get_schema(),
        target_size_gb=partition_size_gb, rows_per_rowgroup=rows_per_rowgroup,
        table_factory=create_pyarrow_table,
    )

    embedding_dim: Optional[int] = None
    batch_inputs: List[str] = []
    batch_metadata: List[Dict] = []

    def _flush_batch():
        nonlocal embedding_dim
        embeddings_np = generator.encode_batch(batch_inputs).cpu().float().numpy().astype(np.float32)
        if embedding_dim is None:
            embedding_dim = embeddings_np.shape[1]
            logger.info("Embedding dimension: %d", embedding_dim)
        writer.add_batch(batch_metadata, embeddings_np, embedding_dim)

    try:
        with tqdm(desc="reviews", unit="rev") as pbar:
            for input_file in input_files:
                logger.info("Processing %s", input_file.name)
                for record in stream_reviews_from_jsonl_gz(input_file):
                    title = record.get('title', '').strip()
                    text  = record.get('text', '').strip()
                    batch_inputs.append(f"{title}. {text}" if title else text)
                    batch_metadata.append({
                        'asin':              record.get('asin'),
                        'parent_asin':       record.get('parent_asin'),
                        'user_id':           record.get('user_id'),
                        'timestamp':         record.get('timestamp'),
                        'rating':            record.get('rating'),
                        'helpful_vote':      record.get('helpful_vote'),
                        'verified_purchase': record.get('verified_purchase'),
                        'title':             title,
                        'text':              text,
                    })

                    if len(batch_inputs) >= batch_size:
                        _flush_batch()
                        pbar.update(len(batch_inputs))
                        batch_inputs, batch_metadata = [], []

            if batch_inputs:
                _flush_batch()
                pbar.update(len(batch_inputs))

    except KeyboardInterrupt:
        logger.warning("Interrupted")
        if embedding_dim is not None:
            writer.close(embedding_dim)
        sys.exit(1)

    except Exception:
        logger.exception("Error during processing")
        if embedding_dim is not None:
            writer.close(embedding_dim)
        raise

    if embedding_dim is not None:
        writer.close(embedding_dim)

    partition_files = sorted(output_dir.glob("part*.parquet"))
    total_size = sum(f.stat().st_size for f in partition_files)
    logger.info("Done. dim=%d | %d partitions | %.1f MB total",
                embedding_dim or 0, len(partition_files), total_size / 1024 / 1024)


def main():
    args = parse_args()
    process_reviews(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        model_size=args.model_size,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        use_fp16=not args.no_fp16,
        partition_size_gb=args.partition_size_gb,
        rows_per_rowgroup=args.rows_per_rowgroup,
    )


if __name__ == "__main__":
    main()
