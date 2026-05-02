#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sys

def convert_to_parquet(input_file: str, output_file: str, datasets_to_extract: list[str], limit, verbose: bool):
    """
    Reads specified datasets from an HDF5 file, validates them, and writes
    them to a single Parquet file.

    Args:
        input_file (str): Path to the source HDF5 file.
        output_file (str): Path to the destination Parquet file.
        datasets_to_extract (list[str]): A list of dataset names to include in the Parquet file.
        limit (int | None): The maximum number of rows to process. If None, all rows are processed.
        verbose (bool): If True, prints detailed information during the conversion.
    """
    print(f"Opening HDF5 file: {input_file}")
    try:
        with h5py.File(input_file, 'r') as hf:
            # --- 1. Validation Step ---
            # Ensure all requested datasets exist and have consistent row counts.
            print("Validating datasets...")
            if not datasets_to_extract:
                raise ValueError("No datasets specified for extraction.")

            first_dataset_name = datasets_to_extract[0]
            if first_dataset_name not in hf:
                raise FileNotFoundError(f"The first specified dataset '{first_dataset_name}' was not found in the HDF5 file.")

            # Determine the total number of rows from the first dataset
            num_rows = hf[first_dataset_name].shape[0]

            # Check other datasets for consistency
            for name in datasets_to_extract[1:]:
                if name not in hf:
                    raise FileNotFoundError(f"Dataset '{name}' was not found in the HDF5 file.")
                if hf[name].shape[0] != num_rows:
                    raise ValueError(
                        "Inconsistent number of rows detected. "
                        f"Dataset '{first_dataset_name}' has {num_rows} rows, but "
                        f"dataset '{name}' has {hf[name].shape[0]} rows."
                    )
            
            print(f"All {len(datasets_to_extract)} specified datasets found with a consistent row count of {num_rows}.")

            # Apply the row limit if provided
            total_rows = num_rows
            if limit is not None:
                if limit <= 0:
                    raise ValueError("Limit must be a positive integer.")
                total_rows = min(num_rows, limit)
                print(f"Applying limit: will process the first {total_rows} rows.")

            # --- 2. Data Preparation Step ---
            # Prepare lists to hold Arrow arrays and schema fields.
            arrow_arrays = []
            schema_fields = []

            # Create and add the 'id' column, which is not in the source file.
            print("Generating 'id' column...")
            id_array = pa.array(np.arange(total_rows), type=pa.int32())
            arrow_arrays.append(id_array)
            schema_fields.append(pa.field('id', id_array.type, nullable=False))
            
            # --- 3. Dataset Processing Step ---
            # Iterate through each dataset, convert it to an Arrow Array, and create its schema field.
            for name in datasets_to_extract:
                dset = hf[name]
                if verbose:
                    print(f"Processing dataset '{name}' (shape: {dset.shape}, dtype: {dset.dtype})...")
                
                # Read the data slice from the HDF5 file. This is memory-efficient.
                data_slice = dset[:total_rows]

                if dset.ndim == 1 or dset.shape[1] == 1:
                    # Case 1: The dataset is a 1D array (a single column).
                    arrow_array = pa.array(data_slice.flatten())
                    field = pa.field(name, arrow_array.type)
                    arrow_arrays.append(arrow_array)
                    schema_fields.append(field)

                elif dset.ndim == 2:
                    # Check for the special case where the dataset name ends with '_label'
                    if name.endswith('_label'):
                        # Case 2a: Multi-column label dataset. Unpack into separate columns.
                        num_label_cols = dset.shape[1]
                        if verbose:
                            print(f"  -> Detected label dataset '{name}'. Unpacking into {num_label_cols} separate columns.")
                        
                        # Iterate through each column of the source matrix
                        for i in range(num_label_cols):
                            col_name = f"{name}_{i}"
                            col_data = data_slice[:, i]
                            
                            # Create and append the Arrow array and schema field for this single column
                            arrow_array = pa.array(col_data)
                            field = pa.field(col_name, arrow_array.type)
                            arrow_arrays.append(arrow_array)
                            schema_fields.append(field)
                    else:
                        # Case 2b: Standard 2D dataset (e.g., vectors).
                        # This will be represented as a fixed-size list in Arrow.
                        element_type = pa.from_numpy_dtype(dset.dtype)
                        list_length = dset.shape[1]
                        field_type = pa.list_(element_type, list_length)
                        
                        # For list types, PyArrow expects a list of lists.
                        arrow_array = pa.array(data_slice.tolist(), type=field_type)
                        field = pa.field(name, field_type)
                        arrow_arrays.append(arrow_array)
                        schema_fields.append(field)
                else:
                    # Case 3: Unsupported shape.
                    raise TypeError(
                        f"Dataset '{name}' has {dset.ndim} dimensions. "
                        "This script only supports 1D and 2D datasets."
                    )
            
            # --- 4. Table Creation and Writing Step ---
            # Construct the Arrow schema and table.
            schema = pa.schema(schema_fields)
            table = pa.Table.from_arrays(arrow_arrays, schema=schema)

            if verbose:
                print("\n--- Verbose Output ---")
                print("Generated Arrow Schema:")
                print(schema)
                print("\nFirst 5 rows of the generated table:")
                print(table.slice(length=5))
                print("----------------------\n")

            print(f"Writing {table.num_rows} rows and {table.num_columns} columns to Parquet file: {output_file}")
            pq.write_table(table, output_file)

    except FileNotFoundError as e:
        print(f"Error: Input file not found at '{input_file}'.", file=sys.stderr)
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert specified datasets from an HDF5 file to a single Parquet file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python %(prog)s my_data.h5 my_data.parquet train_vec train_label --limit 10000 -v"
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
        help="One or more names of datasets to extract from the HDF5 file (e.g., train_vec train_label)."
    )
    parser.add_argument(
        "-l", "--limit", 
        type=int, 
        default=None, 
        help="Limit to the first N rows. If not set, all rows are processed."
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose output, showing schema and sample data."
    )

    args = parser.parse_args()

    try:
        convert_to_parquet(
            input_file=args.input_file,
            output_file=args.output_file,
            datasets_to_extract=args.datasets,
            limit=args.limit,
            verbose=args.verbose
        )
        print("\nConversion process finished successfully.")
    except (ValueError, TypeError, FileNotFoundError) as e:
        print(f"\nERROR: Conversion failed. {e}", file=sys.stderr)
        sys.exit(1)