import duckdb
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to generate TPC-H data for a specific scale factor."
    )
    parser.add_argument(
        "--sf", type=float, default=0.001, help="Scale factor (default: 0.001)"
    )
    parser.add_argument(
        "--n_files",
        type=int,
        default=1,
        help="How many files to split the tables into (default: 1)",
    )
    parser.add_argument(
        "--output_dir_parquet",
        type=str,
        default=None,
        help="Where to write the output files",
    )
    parser.add_argument(
        "--output_dir_csv",
        type=str,
        default=None,
        help="Where to write CSV output files",
    )
    parser.add_argument(
        "--output_dir_tbl",
        type=str,
        default=None,
        help="Where to write TPC-H .tbl output files (pipe-delimited)",
    )
    args = parser.parse_args()
    if args.output_dir_parquet is None and args.output_dir_csv is None and args.output_dir_tbl is None:
        parser.error("At least one of --output_dir_parquet, --output_dir_csv, or --output_dir_tbl must be provided.")
    return args


def main():
    args = parse_args()

    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={args.sf})")

    tables = [
        "customer",
        "lineitem",
        "nation",
        "orders",
        "part",
        "partsupp",
        "region",
        "supplier",
    ]

    n_files = args.n_files

    parquet_dir = Path(args.output_dir_parquet) if args.output_dir_parquet else None
    csv_dir = Path(args.output_dir_csv) if args.output_dir_csv else None
    tbl_dir = Path(args.output_dir_tbl) if args.output_dir_tbl else None

    if parquet_dir:
        parquet_dir.mkdir(parents=True, exist_ok=True)
    if csv_dir:
        csv_dir.mkdir(parents=True, exist_ok=True)
    if tbl_dir:
        tbl_dir.mkdir(parents=True, exist_ok=True)

    for t in tables:
        if n_files == 1:
            if parquet_dir:
                con.execute(f"COPY {t} TO '{parquet_dir / t}.parquet' (FORMAT PARQUET)")
            if csv_dir:
                con.execute(f"COPY {t} TO '{csv_dir / t}.csv' (FORMAT CSV, HEADER)")
            if tbl_dir:
                con.execute(f"COPY {t} TO '{tbl_dir / t}.tbl' (FORMAT CSV, DELIMITER '|', HEADER false)")
        else:
            total_rows = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            batch_size = total_rows // n_files
            for i in range(n_files):
                offset = i * batch_size
                limit = batch_size if i < n_files - 1 else total_rows - offset
                query = f"SELECT * FROM {t} LIMIT {limit} OFFSET {offset}"
                if parquet_dir:
                    con.execute(f"COPY ({query}) TO '{parquet_dir / f'{t}_{i}'}.parquet' (FORMAT PARQUET)")
                if csv_dir:
                    con.execute(f"COPY ({query}) TO '{csv_dir / f'{t}_{i}'}.csv' (FORMAT CSV, HEADER)")
                if tbl_dir:
                    con.execute(f"COPY ({query}) TO '{tbl_dir / f'{t}_{i}'}.tbl' (FORMAT CSV, DELIMITER '|', HEADER false)")


if __name__ == "__main__":
    main()
