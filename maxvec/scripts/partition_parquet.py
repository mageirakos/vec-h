#!/usr/bin/env python3
"""
Partition a single Parquet file into N equal-row-range fragments.

Usage:
    reviews: python ./scripts/partition_parquet.py --input tests/vsds/data-industrial_and_scientific-sf_1/reviews.parquet --n_partitions 10
    images : python ./scripts/partition_parquet.py --input tests/vsds/data-industrial_and_scientific-sf_1/images.parquet --n_partitions 10

The output files are named `<tablename>-0.parquet`, `<tablename>-1.parquet`, etc.
and are written to the same directory as the input file by default.
"""

import argparse
import math
import os
import sys

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a Parquet file into N equal-row-range partition files."
    )
    parser.add_argument("--input", required=True, help="Path to the input .parquet file")
    parser.add_argument(
        "--n_partitions",
        type=int,
        default=None,
        help="Number of output partitions (mutually exclusive with --target_size_gb)",
    )
    parser.add_argument(
        "--target_size_gb",
        type=float,
        default=1.0,
        help="Target size per partition in GB; used to auto-compute N when "
             "--n_partitions is not given (default: 1.0)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for partition files (default: same directory as input)",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Determine base name (e.g. "reviews" from "reviews.parquet")
    basename = os.path.basename(input_path)
    if not basename.endswith(".parquet"):
        print(f"ERROR: input file must have a .parquet extension: {basename}", file=sys.stderr)
        sys.exit(1)
    table_name = basename[: -len(".parquet")]

    # Output directory
    output_dir = args.output_dir if args.output_dir else os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)

    # Determine number of partitions
    if args.n_partitions is not None:
        n_partitions = args.n_partitions
        if n_partitions < 1:
            print("ERROR: --n_partitions must be >= 1", file=sys.stderr)
            sys.exit(1)
    else:
        file_size_bytes = os.path.getsize(input_path)
        file_size_gb = file_size_bytes / (1024 ** 3)
        n_partitions = max(1, math.ceil(file_size_gb / args.target_size_gb))
        print(
            f"File size: {file_size_gb:.2f} GB  →  auto-computed N = {n_partitions} "
            f"(target {args.target_size_gb:.1f} GB/partition)"
        )

    print(f"Reading {input_path} …")
    table = pq.read_table(input_path)
    total_rows = table.num_rows
    print(f"  Total rows: {total_rows:,}  |  Schema: {table.schema}")

    rows_per_part = math.ceil(total_rows / n_partitions)
    print(f"  Partitions: {n_partitions}  |  Rows per partition: ~{rows_per_part:,}")

    for i in range(n_partitions):
        start = i * rows_per_part
        length = min(rows_per_part, total_rows - start)
        if length <= 0:
            print(f"  Partition {i}: skipped (no rows)")
            continue

        part = table.slice(start, length)
        out_path = os.path.join(output_dir, f"{table_name}-{i}.parquet")
        print(f"  Writing partition {i}: rows [{start:,}, {start + length:,}) → {out_path}")
        pq.write_table(part, out_path)

    # --- New Logic to handle the original file automatically ---
    bak_dir = os.path.join(os.path.dirname(input_path), "bak")
    
    print(f"\nMoving original file to: {bak_dir} ...")
    os.makedirs(bak_dir, exist_ok=True)
    
    try:
        import shutil
        dest_path = os.path.join(bak_dir, basename)
        shutil.move(input_path, dest_path)
        print(f"Successfully moved original file to prevent duplicate indexing.")
    except Exception as e:
        print(f"WARNING: Could not move original file: {e}")

    print(f"\nDone. {n_partitions} partition file(s) written to: {output_dir}")


if __name__ == "__main__":
    main()
