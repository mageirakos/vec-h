#!/usr/bin/env python3
import argparse
import sys
import os
import pandas as pd


def load_and_print_head(path: str, n: int = 5) -> None:
    """Load a parquet file from `path` and print the head (n rows).

    Exits with non-zero status if the file cannot be found or read.
    """
    if path == "":
        return
    if not os.path.exists(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"Failed to read parquet file '{path}': {e}", file=sys.stderr)
        sys.exit(1)
    
    print("\n")
    print(f"Loaded parquet: {path} (rows={len(df)})")
    print("\n")
    print(df.head(n))


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a Parquet file and print its head")
    parser.add_argument(
        "path",
        nargs="?",
        default="",
        # default="./tests/vsds/data-industrial_and_scientific-sf_0.01/images.parquet",
        # default="./tests/vsds/data-industrial_and_scientific-sf_0.01/reviews_queries.parquet",
        # default="./tests/vsds/data-industrial_and_scientific-sf_0.01/reviews.parquet",
        # default="./tests/vsds/data-industrial_and_scientific-sf_0.01/images_queries.parquet",

        # default="./tests/vsds/data-industrial_and_scientific-sf_1/images.parquet",
        # default="./tests/vsds/data-industrial_and_scientific-sf_1/reviews_queries.parquet",
        # default="./tests/vsds/data-industrial_and_scientific-sf_1/reviews.parquet",
        # default="./tests/vsds/data-industrial_and_scientific-sf_1/images_queries.parquet",

        # sf 1
        # raw images
        # default="./tests/vsds/data-industrial_and_scientific-sf_1/raw/industrial_and_scientific_sf1_images_queries.parquet",
        # raw reviews
        # default="./tests/vsds/data-industrial_and_scientific-sf_1/raw/industrial_and_scientific_sf1_reviews_queries.parquet",
        

        # sf 0.01
        # raw images
        # default="./tests/vsds/data-industrial_and_scientific-sf_0.01/raw/industrial_and_scientific_sf0.01_images_queries.parquet",
        # raw reviews
        # default="./tests/vsds/data-industrial_and_scientific-sf_0.01/raw/industrial_and_scientific_sf0.01_reviews_queries.parquet",

        # OTHER
        # default="/local/home/vmageirakos/datasets/amazon-23/final_parquet/industrial_and_scientific_sf1_reviews.parquet",
        # default="/local/home/vmageirakos/datasets/amazon-23/final_parquet/industrial_and_scientific_sf1_images.parquet",
        
        
        help=(
            "Path to the parquet file'"
        ),
    )
    parser.add_argument("-n", "--num", type=int, default=5, help="Number of rows to show")
    args = parser.parse_args()

    load_and_print_head(args.path, args.num)

    print("\n\n--- Print SF1 ---\n\n")
    load_and_print_head("./tests/vsds/data-industrial_and_scientific-sf_1/images.parquet", args.num)
    load_and_print_head("./tests/vsds/data-industrial_and_scientific-sf_1/reviews.parquet", args.num)


if __name__ == "__main__":
    main()

