#!/usr/bin/env python3
"""
Load TPC-H + Reviews/Images data into PostgreSQL with pgvector.

Usage examples:
  # Full load with TPC-H from csv files + reviews/images from parquet
  python load_to_postgres_database.py --sf 1 --tpch_dir ~/datasets/csv-1 \
      --reviews_file ~/datasets/amazon-23/final_parquet/all_beauty_sf1_reviews.parquet \
      --images_file ~/datasets/amazon-23/final_parquet/all_beauty_sf1_images.parquet

  # Drop database first (full reset)
  python load_to_postgres_database.py --sf 1 --drop_db ...

  # Skip existing tables (incremental load)
  python load_to_postgres_database.py --sf 1 --skip_existing ...
"""

import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.csv as pacsv
import psycopg
from pgvector.psycopg import register_vector
from tqdm import tqdm
from typing import Dict, List, Optional
import numpy as np
import argparse
from pathlib import Path
import signal
import sys

# Global connection reference for signal handler
_current_connection = None

# ============================================================================
# TPC-H Schema Definition
# ============================================================================

TPCH_TABLES = ["region", "nation", "part", "supplier", "partsupp", "customer", "orders", "lineitem"]

TPCH_SCHEMA: Dict[str, List[str]] = {
    "region": ["r_regionkey", "r_name", "r_comment"],
    "nation": ["n_nationkey", "n_name", "n_regionkey", "n_comment"],
    "part": ["p_partkey", "p_name", "p_mfgr", "p_brand", "p_type", "p_size", "p_container", "p_retailprice", "p_comment"],
    "supplier": ["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"],
    "partsupp": ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"],
    "customer": ["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal", "c_mktsegment", "c_comment"],
    "orders": ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate", "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"],
    "lineitem": ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus", "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipinstruct", "l_shipmode", "l_comment"],
}

# TPC-H table creation DDL (PostgreSQL compatible)
TPCH_DDL = """
CREATE TABLE IF NOT EXISTS region (
    r_regionkey INTEGER PRIMARY KEY,
    r_name CHAR(25),
    r_comment VARCHAR(152)
);

CREATE TABLE IF NOT EXISTS nation (
    n_nationkey INTEGER PRIMARY KEY,
    n_name CHAR(25),
    n_regionkey INTEGER REFERENCES region(r_regionkey),
    n_comment VARCHAR(152)
);

CREATE TABLE IF NOT EXISTS part (
    p_partkey INTEGER PRIMARY KEY,
    p_name VARCHAR(55),
    p_mfgr CHAR(25),
    p_brand CHAR(10),
    p_type VARCHAR(25),
    p_size INTEGER,
    p_container CHAR(10),
    p_retailprice DECIMAL(15,2),
    p_comment VARCHAR(23)
);

CREATE TABLE IF NOT EXISTS supplier (
    s_suppkey INTEGER PRIMARY KEY,
    s_name CHAR(25),
    s_address VARCHAR(40),
    s_nationkey INTEGER REFERENCES nation(n_nationkey),
    s_phone CHAR(15),
    s_acctbal DECIMAL(15,2),
    s_comment VARCHAR(101)
);

CREATE TABLE IF NOT EXISTS partsupp (
    ps_partkey INTEGER REFERENCES part(p_partkey),
    ps_suppkey INTEGER REFERENCES supplier(s_suppkey),
    ps_availqty INTEGER,
    ps_supplycost DECIMAL(15,2),
    ps_comment VARCHAR(199),
    PRIMARY KEY (ps_partkey, ps_suppkey)
);

CREATE TABLE IF NOT EXISTS customer (
    c_custkey INTEGER PRIMARY KEY,
    c_name VARCHAR(25),
    c_address VARCHAR(40),
    c_nationkey INTEGER REFERENCES nation(n_nationkey),
    c_phone CHAR(15),
    c_acctbal DECIMAL(15,2),
    c_mktsegment CHAR(10),
    c_comment VARCHAR(117)
);

CREATE TABLE IF NOT EXISTS orders (
    o_orderkey INTEGER PRIMARY KEY,
    o_custkey INTEGER REFERENCES customer(c_custkey),
    o_orderstatus CHAR(1),
    o_totalprice DECIMAL(15,2),
    o_orderdate DATE,
    o_orderpriority CHAR(15),
    o_clerk CHAR(15),
    o_shippriority INTEGER,
    o_comment VARCHAR(79)
);

CREATE TABLE IF NOT EXISTS lineitem (
    l_orderkey INTEGER REFERENCES orders(o_orderkey),
    l_partkey INTEGER,
    l_suppkey INTEGER,
    l_linenumber INTEGER,
    l_quantity DECIMAL(15,2),
    l_extendedprice DECIMAL(15,2),
    l_discount DECIMAL(15,2),
    l_tax DECIMAL(15,2),
    l_returnflag CHAR(1),
    l_linestatus CHAR(1),
    l_shipdate DATE,
    l_commitdate DATE,
    l_receiptdate DATE,
    l_shipinstruct CHAR(25),
    l_shipmode CHAR(10),
    l_comment VARCHAR(44),
    PRIMARY KEY (l_orderkey, l_linenumber),
    FOREIGN KEY (l_partkey, l_suppkey) REFERENCES partsupp(ps_partkey, ps_suppkey)
);
"""

# ============================================================================
# Database Connection & Utilities
# ============================================================================

def get_db_connection(db_name: str, db_user: str, db_password: str, db_host: str, db_port: str, register_vec: bool = True):
    """Establishes a connection to the PostgreSQL database with optional pgvector support."""
    conn = psycopg.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
    )
    if register_vec:
        register_vector(conn)
    return conn


def get_maintenance_connection(db_user: str, db_password: str, db_host: str, db_port: str):
    """Connect to 'postgres' database for administrative operations."""
    conn = psycopg.connect(
        dbname="postgres",
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port,
        autocommit=True
    )
    return conn


def database_exists(conn, db_name: str) -> bool:
    """Check if database exists."""
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
    return cur.fetchone() is not None


def table_exists(conn, table_name: str) -> bool:
    """Check if table exists and has rows."""
    cur = conn.cursor()
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = %s
        )
    """, (table_name,))
    return cur.fetchone()[0]


def table_row_count(conn, table_name: str) -> int:
    """Get row count of a table."""
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cur.fetchone()[0]
    except:
        return 0


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully by rolling back transaction and closing connection."""
    print("\n\nInterrupt received! Rolling back transaction and closing connection...")
    if _current_connection:
        try:
            _current_connection.rollback()
            _current_connection.close()
        except:
            pass
    sys.exit(1)


# ============================================================================
# Reviews/Images Table DDL
# ============================================================================

def create_vector_table_sql(table_name: str, vector_dim: int) -> str:
    """Generates the CREATE TABLE SQL for reviews/images tables."""
    # Handle _queries suffix for column names
    is_query_table = table_name.endswith("_queries")
    suffix = "_queries" if is_query_table else ""
    
    if table_name.startswith("reviews"):
        return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            rv_reviewkey{suffix} INTEGER PRIMARY KEY,
            rv_rating{suffix} REAL,
            rv_helpful_vote{suffix} INTEGER,
            rv_title{suffix} TEXT,
            rv_text{suffix} TEXT,
            rv_embedding{suffix} VECTOR({vector_dim}),
            rv_partkey{suffix} INTEGER,
            rv_custkey{suffix} INTEGER
        );
        """
    elif table_name.startswith("images"):
        return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            i_imagekey{suffix} INTEGER PRIMARY KEY,
            i_image_url{suffix} TEXT,
            i_variant{suffix} VARCHAR(255),
            i_embedding{suffix} VECTOR({vector_dim}),
            i_partkey{suffix} INTEGER
        );
        """
    else:
        raise ValueError(f"Unknown table name: {table_name}")


def add_fk_constraints_sql(table_name: str) -> str:
    """Add FK constraints for reviews/images tables AFTER data is loaded."""
    if table_name == "reviews":
        return """
        ALTER TABLE reviews ADD CONSTRAINT fk_rv_partkey FOREIGN KEY (rv_partkey) REFERENCES part(p_partkey);
        ALTER TABLE reviews ADD CONSTRAINT fk_rv_custkey FOREIGN KEY (rv_custkey) REFERENCES customer(c_custkey);
        """
    elif table_name == "images":
        return """
        ALTER TABLE images ADD CONSTRAINT fk_i_partkey FOREIGN KEY (i_partkey) REFERENCES part(p_partkey);
        """
    # Query tables don't need FK constraints
    return ""


# ============================================================================
# TPC-H CSV Loading
# ============================================================================

def load_tpch_csv(conn, tpch_dir: Path, skip_existing: bool = False, use_server_side_copy: bool = False):
    """Load TPC-H tables from CSV files."""
    cur = conn.cursor()
    
    # Create schema
    print("Creating TPC-H schema...")
    cur.execute(TPCH_DDL)
    conn.commit()
    
    # Load tables in order (respecting FK dependencies)
    for table_name in TPCH_TABLES:
        csv_path = tpch_dir / f"{table_name}.csv"
        
        if not csv_path.exists():
            print(f"  Warning: {csv_path} not found, skipping {table_name}")
            continue
        
        # Check if table has data
        if skip_existing and table_row_count(conn, table_name) > 0:
            print(f"  Skipping {table_name} (already has {table_row_count(conn, table_name):,} rows)")
            continue
        
        print(f"  Loading {table_name} from {csv_path}...")
        
        if use_server_side_copy:
            # Optimize: Load directly from file (server-side COPY)
            # We assume the standard Docker mount: ~/datasets -> /datasets
            container_path = str(csv_path).replace(str(Path.home() / "datasets"), "/datasets")
            try:
                sql = f"COPY {table_name} FROM '{container_path}' WITH (FORMAT CSV, HEADER, DELIMITER ',')"
                cur.execute(sql)
                conn.commit()
                print(f"    Loaded successfully")
                continue
            except Exception as e:
                print(f"    Server-side copy failed ({e}), falling back to client-side...")
                conn.rollback()

        # Client-side COPY using raw CSV streaming (fast)
        columns = TPCH_SCHEMA[table_name]
        copy_sql = f"COPY {table_name} ({', '.join(columns)}) FROM STDIN WITH (FORMAT CSV)"
        
        total_rows = 0
        with open(csv_path, 'r') as f:
            header = f.readline()  # skip CSV header
            with cur.copy(copy_sql) as copy:
                while True:
                    chunk = f.read(8 * 1024 * 1024)  # 8MB chunks
                    if not chunk:
                        break
                    copy.write(chunk.encode('utf-8'))
        
        conn.commit()
        count = table_row_count(conn, table_name)
        print(f"    Loaded {count:,} rows")
    
    print("TPC-H loading complete.")


# ============================================================================
# Parquet Loading (Reviews/Images)
# ============================================================================

def load_parquet_to_db(
    file_path: str, 
    table_name: str, 
    vector_dim: int, 
    target_dim: Optional[int] = None,
    db_name: str = "vech", 
    db_user: str = "postgres", 
    db_password: str = "1234",
    db_host: str = "localhost", 
    db_port: str = "5432", 
    set_storage_plain: bool = False,
    skip_existing: bool = False,
    add_fk: bool = True
):
    """Load a reviews/images parquet file into PostgreSQL."""
    global _current_connection
    
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        print(f"Error: File not found at {file_path}")
        return False

    # Determine actual dimension to use
    db_vector_dim = target_dim if target_dim else vector_dim
    print(f"Loading data from {file_path} into table {table_name}...", flush=True)
    if target_dim and target_dim < vector_dim:
        print(f"  Truncating vectors from {vector_dim} to {target_dim} dimensions", flush=True)

    # Open Parquet file (metadata only, no data loaded yet)
    try:
        parquet_file = pq.ParquetFile(file_path)
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        return False

    column_names = [field.name for field in parquet_file.schema_arrow]
    num_rows = parquet_file.metadata.num_rows
    num_row_groups = parquet_file.metadata.num_row_groups
    print(f"Total rows: {num_rows:,} ({num_row_groups} row groups)")
    
    # Connect
    conn = None
    try:
        conn = get_db_connection(db_name, db_user, db_password, db_host, db_port)
        _current_connection = conn
        cur = conn.cursor()
        
        # Check if table exists with data
        if skip_existing and table_exists(conn, table_name):
            count = table_row_count(conn, table_name)
            if count > 0:
                print(f"  Skipping {table_name} (already has {count:,} rows)")
                return True
        
        # Disable synchronous_commit for faster inserts
        cur.execute("SET synchronous_commit = off;")

        # Drop existing table if it exists
        print(f"Dropping existing {table_name} table if it exists...")
        cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
        
        # Create Table
        print(f"Creating table {table_name} with vector dimension {db_vector_dim}...", flush=True)
        cur.execute(create_vector_table_sql(table_name, db_vector_dim))
        
        # Set storage plain if requested
        if set_storage_plain:
            if table_name.startswith('reviews'):
                suffix = '_queries' if table_name.endswith('_queries') else ''
                vec_col = f'rv_embedding{suffix}'
            elif table_name.startswith('images'):
                suffix = '_queries' if table_name.endswith('_queries') else ''
                vec_col = f'i_embedding{suffix}'
            else:
                raise ValueError(f"Unknown table name: {table_name}")
            cur.execute(f"ALTER TABLE {table_name} ALTER COLUMN {vec_col} SET STORAGE PLAIN")
        conn.commit()
        
        # COPY Setup
        copy_sql = f"COPY {table_name} ({', '.join(column_names)}) FROM STDIN WITH (FORMAT BINARY)"
        batch_size = 10000
        
        print(f"Streaming {num_rows:,} rows (batch_size={batch_size})...", flush=True)

        # Build column types for binary COPY
        def get_column_types(table_name: str, column_names: list) -> list:
            type_map = {
                # Reviews table columns
                'rv_reviewkey': 'int4', 'rv_reviewkey_queries': 'int4',
                'rv_rating': 'float4', 'rv_rating_queries': 'float4',
                'rv_helpful_vote': 'int4', 'rv_helpful_vote_queries': 'int4',
                'rv_title': 'text', 'rv_title_queries': 'text',
                'rv_text': 'text', 'rv_text_queries': 'text',
                'rv_embedding': 'vector', 'rv_embedding_queries': 'vector',
                'rv_partkey': 'int4', 'rv_partkey_queries': 'int4',
                'rv_custkey': 'int4', 'rv_custkey_queries': 'int4',
                # Images table columns
                'i_imagekey': 'int4', 'i_imagekey_queries': 'int4',
                'i_image_url': 'text', 'i_image_url_queries': 'text',
                'i_variant': 'text', 'i_variant_queries': 'text',
                'i_embedding': 'vector', 'i_embedding_queries': 'vector',
                'i_partkey': 'int4', 'i_partkey_queries': 'int4',
            }
            return [type_map[col] for col in column_names]
        
        column_types = get_column_types(table_name, column_names)

        # Stream row groups from parquet file (only one row group in memory at a time)
        rows_inserted = 0
        with tqdm(total=num_rows) as pbar:
            for rg_idx in range(num_row_groups):
                rg_table = parquet_file.read_row_group(rg_idx)
                rg_num_rows = rg_table.num_rows
                num_batches = (rg_num_rows + batch_size - 1) // batch_size

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, rg_num_rows)
                    
                    batch_arrow = rg_table.slice(start_idx, end_idx - start_idx)
                    df_batch = batch_arrow.to_pandas()
                    
                    try:
                        with cur.copy(copy_sql) as copy:
                            copy.set_types(column_types)
                            
                            for _, row in df_batch.iterrows():
                                row_values = []
                                for col_name in column_names:
                                    val = row[col_name]
                                    if val is None or (isinstance(val, float) and np.isnan(val)):
                                        row_values.append(None)
                                    elif col_name.endswith('_embedding') or col_name.endswith('_embedding_queries'):
                                        vec = np.asarray(val, dtype=np.float32)
                                        if target_dim and target_dim < vector_dim:
                                            vec = vec[:target_dim]
                                        row_values.append(vec)
                                    elif isinstance(val, str):
                                        # Strip null bytes — PostgreSQL rejects \x00 in text
                                        row_values.append(val.replace('\x00', ''))
                                    else:
                                        row_values.append(val)
                                
                                copy.write_row(row_values)
                        
                        batch_rows = end_idx - start_idx
                        rows_inserted += batch_rows
                        pbar.update(batch_rows)
                        conn.commit()
                    except Exception as e:
                        print(f"\n--- FATAL ERROR IN ROW GROUP {rg_idx}, BATCH {batch_idx+1} ---")
                        print(f"Database Error: {e}")
                        conn.rollback()
                        raise

                # Free row group memory before reading next one
                del rg_table
        
        print(f"Data insertion completed. Total rows inserted: {rows_inserted:,}")

        # Add FK constraints (skip for query tables)
        if add_fk and not table_name.endswith("_queries"):
            fk_sql = add_fk_constraints_sql(table_name)
            if fk_sql:
                print(f"Adding foreign key constraints...", flush=True)
                cur.execute(fk_sql)
        
        conn.commit()
        print(f"Successfully loaded data into {table_name}.")
        return True

    except (Exception, psycopg.Error) as error:
        print(f"Error: {error}")
        if conn:
            conn.rollback()
        return False
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user.")
        if conn:
            conn.rollback()
        raise
    finally:
        _current_connection = None
        if conn and not conn.closed:
            cur.close()
            conn.close()


# ============================================================================
# Main
# ============================================================================

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(
        description="Load TPC-H + Reviews/Images data into PostgreSQL with pgvector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load everything (TPC-H + reviews/images)
  python load_to_postgres_database.py --sf 1 --tpch_dir ~/datasets/csv-1 \\
      --reviews_file ~/datasets/final_parquet/all_beauty_sf1_reviews.parquet \\
      --images_file ~/datasets/final_parquet/all_beauty_sf1_images.parquet

  # Drop database first (full reset)
  python load_to_postgres_database.py --sf 1 --drop_db ...

  # Only load TPC-H (skip reviews/images)
  python load_to_postgres_database.py --sf 1 --tpch_dir ~/datasets/csv-1 --skip_reviews --skip_images
        """
    )
    
    # Scale factor (used for path defaults)
    parser.add_argument('--sf', type=float, default=1.0, help='Scale factor (for default path construction)')
    
    # Database settings
    parser.add_argument('--db_name', type=str, default="vech", help='Database name (default: vech)')
    parser.add_argument('--db_host', type=str, default="localhost", help='Database host')
    parser.add_argument('--db_port', type=str, default="5432", help='Database port')
    parser.add_argument('--db_user', type=str, default="postgres", help='Database user')
    parser.add_argument('--db_password', type=str, default="1234", help='Database password')
    
    # TPC-H settings
    sf_str = lambda sf: str(int(sf) if float(sf).is_integer() else sf)
    parser.add_argument('--tpch_dir', type=str, default=None,
                       help='Directory containing TPC-H CSV files (default: ~/datasets/tpch-datasets/csv/csv-<sf>)')
    
    # Reviews/Images files
    parser.add_argument('--reviews_file', type=str, default=None,
                       help='Path to reviews parquet file')
    parser.add_argument('--images_file', type=str, default=None,
                       help='Path to images parquet file')
    parser.add_argument('--reviews_queries_file', type=str, default=None,
                       help='Path to reviews_queries parquet file')
    parser.add_argument('--images_queries_file', type=str, default=None,
                       help='Path to images_queries parquet file')
    
    # Vector dimensions
    parser.add_argument('--reviews_vector_dim', type=int, default=1024, 
                       help='Dimension of review embeddings')
    parser.add_argument('--images_vector_dim', type=int, default=1152, 
                       help='Dimension of image embeddings')
    parser.add_argument('--mrl_review_target_dim', type=int, default=None, 
                       help='Target dimension for MRL truncation')
    
    # Control flags
    parser.add_argument('--drop_db', action='store_true', 
                       help='Drop and recreate database (full reset)')
    parser.add_argument('--skip_existing', action='store_true', 
                       help='Skip tables that already have data')
    parser.add_argument('--skip_tpch', action='store_true', help='Skip loading TPC-H tables')
    parser.add_argument('--skip_reviews', action='store_true', help='Skip loading reviews table')
    parser.add_argument('--skip_images', action='store_true', help='Skip loading images table')
    parser.add_argument('--skip_reviews_queries', action='store_true', help='Skip loading reviews_queries')
    parser.add_argument('--skip_images_queries', action='store_true', help='Skip loading images_queries')
    parser.add_argument('--set_storage_plain', action='store_true', 
                       help='Set vector column storage to PLAIN. Otherwise it is TOASTED (EXTENDED)')
    parser.add_argument('--skip_fk', action='store_true', 
                       help='Skip adding foreign key constraints (faster loading)')
    
    parser.add_argument('--use_server_copy', action='store_true', 
        help='Use server-side COPY (faster, requires file access from DB container)')

    args = parser.parse_args()
    
    # Build default paths based on scale factor
    sf = sf_str(args.sf)
    home = Path.home()
    
    if args.tpch_dir is None:
        args.tpch_dir = home / "datasets" / "tpch-datasets" / "csv" / f"csv-{sf}"
    else:
        args.tpch_dir = Path(args.tpch_dir).expanduser()
    
    print("=" * 60)
    print("PostgreSQL Data Loader (TPC-H + VECH)")
    print("=" * 60)
    print(f"Database: {args.db_host}:{args.db_port}/{args.db_name}")
    print(f"TPC-H Directory: {args.tpch_dir}")
    print(f"Reviews file: {args.reviews_file}")
    print(f"Images file: {args.images_file}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Drop DB first: {args.drop_db}")
    print("=" * 60)

    # --- Database Setup ---
    maint_conn = get_maintenance_connection(args.db_user, args.db_password, args.db_host, args.db_port)
    
    if args.drop_db:
        print(f"\nDropping database '{args.db_name}'...")
        maint_conn.execute(f"DROP DATABASE IF EXISTS {args.db_name}")
    
    if not database_exists(maint_conn, args.db_name):
        print(f"Creating database '{args.db_name}'...")
        maint_conn.execute(f"CREATE DATABASE {args.db_name}")
    
    maint_conn.close()
    
    # --- Connect to target database (without vector registration first) ---
    conn = get_db_connection(args.db_name, args.db_user, args.db_password, args.db_host, args.db_port, register_vec=False)
    
    # Enable pgvector extension BEFORE registering vector type
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    
    # Now register the vector type
    register_vector(conn)
    
    # --- Load TPC-H ---
    if not args.skip_tpch:
        print("\n--- Loading TPC-H Tables ---")
        load_tpch_csv(conn, args.tpch_dir, args.skip_existing)
    else:
        print("\nSkipping TPC-H tables (--skip_tpch)")
    
    conn.close()
    
    # --- Load Reviews/Images ---
    print("\n--- Loading Reviews/Images Tables ---")
    
    if args.reviews_file and not args.skip_reviews:
        load_parquet_to_db(
            args.reviews_file, "reviews", args.reviews_vector_dim, args.mrl_review_target_dim,
            args.db_name, args.db_user, args.db_password, args.db_host, args.db_port,
            args.set_storage_plain, args.skip_existing, add_fk=not args.skip_fk
        )
    elif args.skip_reviews:
        print("Skipping reviews table (--skip_reviews)")
    
    if args.images_file and not args.skip_images:
        load_parquet_to_db(
            args.images_file, "images", args.images_vector_dim, None,
            args.db_name, args.db_user, args.db_password, args.db_host, args.db_port,
            args.set_storage_plain, args.skip_existing, add_fk=not args.skip_fk
        )
    elif args.skip_images:
        print("Skipping images table (--skip_images)")
    
    if args.reviews_queries_file and not args.skip_reviews_queries:
        load_parquet_to_db(
            args.reviews_queries_file, "reviews_queries", args.reviews_vector_dim, args.mrl_review_target_dim,
            args.db_name, args.db_user, args.db_password, args.db_host, args.db_port,
            args.set_storage_plain, args.skip_existing, add_fk=not args.skip_fk
        )
    elif args.skip_reviews_queries:
        print("Skipping reviews_queries table (--skip_reviews_queries)")
    
    if args.images_queries_file and not args.skip_images_queries:
        load_parquet_to_db(
            args.images_queries_file, "images_queries", args.images_vector_dim, None,
            args.db_name, args.db_user, args.db_password, args.db_host, args.db_port,
            args.set_storage_plain, args.skip_existing, add_fk=not args.skip_fk
        )
    elif args.skip_images_queries:
        print("Skipping images_queries table (--skip_images_queries)")

    print("\n" + "=" * 60)
    print("Loading completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()