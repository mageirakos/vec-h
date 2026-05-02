#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sys
import math

def convert_to_parquet_chunked(input_file: str, output_file: str, datasets_to_extract: list[str], limit: int | None, chunk_size: int, verbose: bool):
    """
    Reads specified datasets from a large HDF5 file in chunks, validates them, 
    and writes them to a single Parquet file iteratively to keep memory usage low.

    Args:
        input_file (str): Path to the source HDF5 file.
        output_file (str): Path to the destination Parquet file.
        datasets_to_extract (list[str]): A list of dataset names to include in the Parquet file.
        limit (int | None): The maximum number of rows to process. If None, all rows are processed.
        chunk_size (int): The number of rows to process in each chunk.
        verbose (bool): If True, prints detailed information during the conversion.
    """
    print(f"Opening HDF5 file: {input_file}")
    try:
        with h5py.File(input_file, 'r') as hf:
            # --- 1. Validation and Schema Definition Step ---
            print("Validating datasets and preparing schema...")
            if not datasets_to_extract:
                raise ValueError("No datasets specified for extraction.")

            first_dataset_name = datasets_to_extract[0]
            if first_dataset_name not in hf:
                raise FileNotFoundError(f"The first specified dataset '{first_dataset_name}' was not found in the HDF5 file.")

            num_rows = hf[first_dataset_name].shape[0]

            # Determine the total number of rows to process
            total_rows = num_rows
            if limit is not None:
                if limit <= 0:
                    raise ValueError("Limit must be a positive integer.")
                total_rows = min(num_rows, limit)
            
            print(f"Found {num_rows} total rows. Will process {total_rows} rows in chunks of {chunk_size}.")

            # --- Prepare the Arrow schema ahead of time ---
            schema_fields = []
            # Add the 'id' field first
            schema_fields.append(pa.field('id', pa.int64(), nullable=False))

            for name in datasets_to_extract:
                if name not in hf:
                    raise FileNotFoundError(f"Dataset '{name}' was not found in the HDF5 file.")
                dset = hf[name]
                if dset.shape[0] != num_rows:
                    raise ValueError(f"Inconsistent row count in dataset '{name}'.")

                # Determine the field(s) for the schema
                if dset.ndim == 1 or dset.shape[1] == 1:
                    arrow_type = pa.from_numpy_dtype(dset.dtype)
                    schema_fields.append(pa.field(name, arrow_type))
                elif dset.ndim == 2:
                    if name.endswith('_label'):
                        # Unpack each column into a separate field
                        for i in range(dset.shape[1]):
                            col_name = f"{name}_{i}"
                            arrow_type = pa.from_numpy_dtype(dset.dtype)
                            schema_fields.append(pa.field(col_name, arrow_type))
                    else:
                        # Represent as a fixed-size list
                        element_type = pa.from_numpy_dtype(dset.dtype)
                        list_length = dset.shape[1]
                        field_type = pa.list_(element_type, list_length)
                        schema_fields.append(pa.field(name, field_type))
                else:
                    raise TypeError(f"Unsupported dataset dimension ({dset.ndim}) for '{name}'.")

            schema = pa.schema(schema_fields)
            if verbose:
                print("\n--- Verbose Output ---")
                print("Generated Arrow Schema:")
                print(schema)
                print("----------------------\n")

            # --- 2. Chunked Processing and Writing Step ---
            writer = None
            try:
                # Calculate the number of chunks
                num_chunks = math.ceil(total_rows / chunk_size)
                print(f"Starting conversion of {total_rows} rows in {num_chunks} chunks...")

                for i in range(num_chunks):
                    start_row = i * chunk_size
                    end_row = min((i + 1) * chunk_size, total_rows)
                    
                    if start_row >= end_row:
                        continue

                    if verbose:
                        print(f"Processing chunk {i+1}/{num_chunks} (rows {start_row}-{end_row})...")

                    # This list will hold the Arrow Arrays for the current chunk
                    chunk_arrow_arrays = []

                    # Generate 'id' for the chunk
                    id_array = pa.array(np.arange(start_row, end_row), type=pa.int64())
                    chunk_arrow_arrays.append(id_array)

                    # Process each dataset for the current chunk
                    for name in datasets_to_extract:
                        dset = hf[name]
                        data_slice = dset[start_row:end_row]

                        if dset.ndim == 1 or dset.shape[1] == 1:
                            arrow_array = pa.array(data_slice.flatten())
                            chunk_arrow_arrays.append(arrow_array)
                        elif dset.ndim == 2:
                            if name.endswith('_label'):
                                # Unpack each column into a separate Arrow Array
                                for col_idx in range(dset.shape[1]):
                                    col_data = data_slice[:, col_idx]
                                    arrow_array = pa.array(col_data)
                                    chunk_arrow_arrays.append(arrow_array)
                            else:
                                # Create a list array for the vector
                                arrow_array = pa.array(data_slice.tolist())
                                chunk_arrow_arrays.append(arrow_array)
                    
                    # Create a table from the chunk's arrays
                    chunk_table = pa.Table.from_arrays(chunk_arrow_arrays, schema=schema)

                    # Initialize writer on the first chunk
                    if writer is None:
                        writer = pq.ParquetWriter(output_file, schema)
                    
                    # Write the chunk to the Parquet file
                    writer.write_table(chunk_table)

            finally:
                if writer:
                    writer.close()
                    print(f"\nSuccessfully wrote {total_rows} rows and {len(schema)} columns to Parquet file: {output_file}")

    except FileNotFoundError as e:
        print(f"Error: Input file not found at '{input_file}'.", file=sys.stderr)
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert specified datasets from a large HDF5 file to a Parquet file using chunking to manage memory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python %(prog)s big_data.h5 big_data.parquet train_vec train_label --limit 1000000 --chunk-size 50000 -v"
    )
    parser.add_argument(
        "input_file", 
        type=str, 
        help="Path to the input HDF5 file."
    )
    parser.add_argument(
        "output_file", 
        type=str, 
        help="Path for the output Parquet file."
    )
    parser.add_argument(
        "datasets", 
        nargs='+', 
        help="One or more names of datasets to extract from the HDF5 file."
    )
    parser.add_argument(
        "-l", "--limit", 
        type=int, 
        default=None, 
        help="Limit to the first N rows. If not set, all rows are processed."
    )
    parser.add_argument(
        "-c", "--chunk-size", 
        type=int, 
        default=100_000, 
        help="The number of rows to process in each chunk to control memory usage."
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose output, showing schema and progress."
    )

    args = parser.parse_args()

    try:
        convert_to_parquet_chunked(
            input_file=args.input_file,
            output_file=args.output_file,
            datasets_to_extract=args.datasets,
            limit=args.limit,
            chunk_size=args.chunk_size,
            verbose=args.verbose
        )
        print("\nConversion process finished successfully.")
    except (ValueError, TypeError, FileNotFoundError) as e:
        print(f"\nERROR: Conversion failed. {e}", file=sys.stderr)
        sys.exit(1)