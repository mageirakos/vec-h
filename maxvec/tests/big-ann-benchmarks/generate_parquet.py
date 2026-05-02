import argparse
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def read_vectors(file_path, dtype, limit=None, verbose=False):
    with open(file_path, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.uint32)
        num_points, dim = header
        if limit:
            num_points = min(num_points, limit)
        if verbose:
            print(f"Reading {num_points} vectors of dimension {dim} from {file_path} as {dtype}")
        total_values = num_points * dim
        data = np.frombuffer(f.read(total_values * np.dtype(dtype).itemsize), dtype=dtype, count=total_values)
        vectors = data.reshape((num_points, dim))
        return vectors

def read_ground_truth(file_path, limit=None, verbose=False):
    with open(file_path, "rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.uint32)
        num_queries, k = header
        if limit:
            num_queries = min(num_queries, limit)
        if verbose:
            print(f"Reading ground truth for {num_queries} queries with top-{k} neighbors from {file_path}")
        ids = np.frombuffer(f.read(num_queries * k * 4), dtype=np.uint32).reshape((num_queries, k))
        distances = np.frombuffer(f.read(num_queries * k * 4), dtype=np.float32).reshape((num_queries, k))
        return ids, distances

def convert_to_parquet(input_file_path, output_file_path, verbose=False, limit=None):
    ext = Path(input_file_path).suffix.lower()
    
    if ext in [".u8bin", ".fbin", ".i8bin"]:
        dtype = {
            ".u8bin": np.uint8,
            ".fbin": np.float32,
            ".i8bin": np.int8
        }[ext]
        vectors = read_vectors(input_file_path, dtype, limit, verbose)
        dim = vectors.shape[1]
        arrow_type = pa.from_numpy_dtype(dtype)
        list_type = pa.list_(arrow_type, dim)
        data_list = vectors.tolist()
        pa_array = pa.array(data_list, type=list_type)
        table = pa.Table.from_arrays([pa_array], names=["vector"])
        pq.write_table(table, output_file_path)

    elif ext == ".ibin":
        ids, distances = read_ground_truth(input_file_path, limit, verbose)
        k = ids.shape[1]

        ids_array = pa.array(ids.tolist(), type=pa.list_(pa.uint32(), k))
        dists_array = pa.array(distances.tolist(), type=pa.list_(pa.float32(), k))

        table = pa.Table.from_arrays(
            [ids_array, dists_array],
            names=["neighbor_ids", "distances"]
        )
        pq.write_table(table, output_file_path)

    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if verbose:
        print(f"Saved Parquet file to: {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert big-ann-benchmark dataset (.fbin, .u8bin, .i8bin, .ibin) to Parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_file", type=str, help="Path to the input file")
    parser.add_argument("output_file", type=str, help="Path to where the result Parquet file will be saved")
    parser.add_argument("-l", "--limit", type=int, help="Limit to the first N vectors in the input file. Default is to read all vectors.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output, showing details and first few vectors read.")

    args = parser.parse_args()
    convert_to_parquet(args.input_file, args.output_file, args.verbose, args.limit)
    print("\nConversion process finished.")
