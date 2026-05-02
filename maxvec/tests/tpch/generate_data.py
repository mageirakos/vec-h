import duckdb
import pyarrow.parquet as pq
import pyarrow.csv as csv
import argparse
from pathlib import Path

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Script to generate TPC-H data for a specific scale factor.")
    parser.add_argument("--sf", type=float, default=0.001, help="Scale factor (default: 0.001)")
    parser.add_argument("--n_files", type=int, default=1, help="How many files to split the data into (default: 1)")
    return parser.parse_args()

# Parse command-line arguments
args = parse_args()

# Connect to DuckDB
con = duckdb.connect(database=':memory:')

# Load TPCH extension
con.execute("INSTALL tpch; LOAD tpch")

# Call dbgen with the specified scale factor
con.execute(f"CALL dbgen(sf={args.sf})")

print(con.execute("show tables").fetchall())

tables = ["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"]

sf = args.sf
n_files = args.n_files

for t in tables:
    res = con.query("SELECT * FROM " + t)
    arrow_table = res.to_arrow_table()

    if isinstance(sf, int) or sf.is_integer():
        sf = int(sf)

    base_path = "./csv-" + str(sf)
    base_path_pq = "./parquet-" + str(sf)
    Path(base_path).mkdir(parents=True, exist_ok=True)
    Path(base_path_pq).mkdir(parents=True, exist_ok=True)

    pq.write_table(arrow_table, base_path_pq + "/" + t + ".parquet")
    csv.write_csv(arrow_table, base_path + "/" + t + ".csv")

    # split the table into multiple disjoin files
    if n_files > 1:
        for i in range(n_files):
            batch_size = arrow_table.num_rows // n_files
            # we slice the table into n_files disjoint parts
            arrow_table_slice = arrow_table.slice(i * batch_size, batch_size)
            csv.write_csv(arrow_table_slice, base_path + "/" + t + "_" + str(i) + ".csv")

