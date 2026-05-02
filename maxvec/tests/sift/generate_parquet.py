import argparse
import struct
import tarfile
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def read_vecs(filename, extension, verbose=False):
    """
    Reads .fvecs, .ivecs, or .bvecs files.

    Args:
        filename (str or Path): Path to the file.
        extension (str): The file extension ('.fvecs', '.ivecs', '.bvecs').
        verbose (bool): If True, print debug information.

    Returns:
        tuple: (numpy.ndarray, int): The data read from the file and the dimension.
               Returns (None, 0) if the file format is unsupported or read fails.
    """
    if extension == '.fvecs':
        component_type = 'f'
        component_size = 4
        dtype = np.float32
    elif extension == '.ivecs':
        component_type = 'i'
        component_size = 4
        dtype = np.int32
    elif extension == '.bvecs':
        component_type = 'B'
        component_size = 1
        dtype = np.uint8
    else:
        print(f"Error: Unsupported file extension '{extension}' for {filename}")
        return None, 0

    vectors = []
    dim = 0
    try:
        with open(filename, 'rb') as f:
            while True:
                # Read dimension (4 bytes, little endian integer)
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break # End of file
                if len(dim_bytes) != 4:
                    print(f"Warning: Could not read dimension from {filename}. Incomplete read.")
                    break

                dim = struct.unpack('<i', dim_bytes)[0]
                if dim <= 0:
                     print(f"Warning: Invalid dimension {dim} read from {filename}. Skipping vector.")
                     # Try to skip the expected number of bytes if possible, though this might be unreliable
                     # If we don't know the intended dim, we can't reliably skip
                     break # Stop processing this file


                expected_bytes = dim * component_size
                vector_bytes = f.read(expected_bytes)

                if len(vector_bytes) != expected_bytes:
                    print(f"Warning: Could not read vector data from {filename}. Expected {expected_bytes} bytes, got {len(vector_bytes)}. Stopping.")
                    break # Stop processing this file if a vector is incomplete

                # Unpack vector components (little endian)
                fmt = f'<{dim}{component_type}'
                vector = struct.unpack(fmt, vector_bytes)
                vectors.append(vector)

        if not vectors:
             print(f"Warning: No vectors read from {filename}")
             return None, 0

        # Convert list of tuples to a 2D numpy array
        data = np.array(vectors, dtype=dtype)

        if verbose and data.size > 0:
            print(f"\n--- Read {filename} (Extension: {extension}, Type: {dtype}, Shape: {data.shape}) ---")
            print(f"Dimension (d): {dim}")
            print(f"Number of vectors (n): {data.shape[0]}")
            print("First few vectors:")
            print(data[:min(5, data.shape[0])])
            print("-" * (len(str(filename)) + 20))


        return data, dim

    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        return None, 0
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, 0


def convert_to_parquet(input_tar_path, output_dir_path, verbose=False):
    """
    Converts ANN_SIFT dataset files from a tar.gz archive to Parquet files.

    Args:
        input_tar_path (str): Path to the input .tar.gz file (e.g., 'siftsmall.tar.gz').
        output_dir_path (str): Path to the directory where Parquet files will be saved.
        verbose (bool): If True, print detailed processing information.
    """
    input_path = Path(input_tar_path)
    output_path = Path(output_dir_path)

    if not input_path.is_file() or not input_tar_path.endswith('.tar.gz'):
        print(f"Error: Input path '{input_tar_path}' is not a valid .tar.gz file.")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path.resolve()}")

    try:
        with tarfile.open(input_tar_path, 'r:gz') as tar:
            # Extract members to process them
            # Using a temporary directory might be safer in complex scenarios,
            # but processing directly is feasible if member names are trusted.
            # We'll process files directly from the archive stream.

            for member in tar.getmembers():
                if not member.isfile():
                    continue # Skip directories, etc.

                # Extract file extension and base name
                member_path = Path(member.name)
                extension = member_path.suffix.lower()
                base_name = member_path.stem # e.g., 'siftsmall_base'

                if extension not in ['.fvecs', '.ivecs', '.bvecs']:
                    if verbose:
                        print(f"Skipping non-vector file: {member.name}")
                    continue

                print(f"\nProcessing {member.name}...")

                # Extract file content to memory
                try:
                    file_obj = tar.extractfile(member)
                    if file_obj is None:
                        print(f"Warning: Could not extract {member.name} from archive.")
                        continue

                    # Write to a temporary file to use the existing read functions
                    # Alternatively, adapt read_vecs to work with file-like objects
                    temp_file_path = output_path / f"temp_{member_path.name}"
                    with open(temp_file_path, 'wb') as temp_f:
                        temp_f.write(file_obj.read())

                except Exception as e:
                     print(f"Error extracting {member.name}: {e}")
                     continue


                # Read vector data using the appropriate function
                data, dim = read_vecs(temp_file_path, extension, verbose)

                # Clean up temporary file
                try:
                    os.remove(temp_file_path)
                except OSError as e:
                    print(f"Warning: Could not remove temporary file {temp_file_path}: {e}")


                if data is None:
                    print(f"Failed to read data from {member.name}. Skipping Parquet conversion.")
                    continue

                # Prepare data for PyArrow Table
                normalized_base = base_name.split("_", 1)[-1]
                output_parquet_path = output_path / f"{normalized_base}.parquet"
                print(f"Converting {member.name} to {output_parquet_path}...")

                try:
                    # For fvecs/bvecs, store as a single column of lists
                    # Using FixedSizeList is ideal for Arrow compatibility
                    if data.dtype == np.float32:
                        arrow_type = pa.float32()
                    elif data.dtype == np.uint8:
                        arrow_type = pa.uint8()
                    elif data.dtype == np.int32:
                        arrow_type = pa.uint32()
                    else:
                        print(f"Warning: Unexpected numpy dtype {data.dtype} for {member.name}. Defaulting to bytes.")
                        # Fallback or handle other types if needed
                        arrow_type = pa.float32() # Default assumption for safety


                    # Ensure data is 2D
                    if data.ndim == 1:
                        data = data.reshape(-1, dim if dim > 0 else 1) # Reshape if flat

                    if dim > 0:
                        print("Encoding with dim {} and type {}".format(dim, arrow_type))
                        list_type = pa.list_(arrow_type, dim)
                        # Convert numpy array to list of lists for pa.array
                        data_list = data.tolist()
                        pa_array = pa.array(data_list, type=list_type)
                        ids = pa.array(list(range(len(data))), type=pa.int32())
                        table = pa.Table.from_arrays([ids, pa_array], names=['id', 'vector'])
                    else:
                        print(f"Warning: Dimension is 0 for {member.name}. Cannot create fixed size list. Skipping.")
                        continue

                    # Write the PyArrow Table to a Parquet file
                    pq.write_table(table, output_parquet_path)
                    print(f"Successfully wrote {output_parquet_path}")

                except Exception as e:
                    print(f"Error converting or writing Parquet for {member.name}: {e}")


    except FileNotFoundError:
        print(f"Error: Input archive '{input_tar_path}' not found.")
    except tarfile.ReadError:
        print(f"Error: Could not read input archive '{input_tar_path}'. It might be corrupted or not a valid tar.gz file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ANN_SIFT dataset (.fvecs, .ivecs, .bvecs in tar.gz) to Parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_tar",
        type=str,
        help="Path to the input dataset archive (e.g., 'siftsmall.tar.gz')."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the directory where Parquet files will be saved."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output, showing details and first few vectors read."
    )

    args = parser.parse_args()

    convert_to_parquet(args.input_tar, args.output_dir, args.verbose)
    print("\nConversion process finished.")