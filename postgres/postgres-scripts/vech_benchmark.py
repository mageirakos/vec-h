#!/usr/bin/env python3
from __future__ import annotations  # lazy-evaluate annotations so psycopg type hints don't explode when psycopg is absent (parse-only mode)
"""
VECH Benchmark Runner for pgvector.

All VECH query metadata, SQL, embedding loading, execution, and orchestration
in one file. Outputs CSVs in the same format as Maximus's parse_caliper, under
parse_postgres/ so they can be plotted alongside Maximus results.

Usage:

    # Full benchmark (all default queries, 1 warmup + 20 reps, save everything)
    python3 vech_benchmark.py run --sf 1 --index_label none --save_csv --save_plans

    # With an HNSW index (build first, then benchmark with that index_label)
    python3 vech_benchmark.py build-indexes --sf 1 --index hnsw
    python3 vech_benchmark.py run --sf 1 --index_label HNSW32 --save_csv --save_plans

    # Quick one-off (1 rep, no warmup, no flush). Short aliases work: q10 -> q10_mid
    python3 vech_benchmark.py run --sf 1 --query q10 --nreps 1 --no_warmup --no_flush

    # Inspect what indexes exist on reviews/images (auto-derives db_name from --sf)
    python3 vech_benchmark.py check-indexes --sf 1
    # Or explicitly:
    python3 vech_benchmark.py check-indexes --db_name vech_sf1_industrial_and_scientific_plain

    # List available queries
    python3 vech_benchmark.py list

    # Drop an index
    python3 vech_benchmark.py build-indexes --sf 1 --index hnsw --drop
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
# psycopg + pgvector + pyarrow are only needed for live runs. parse-only
# (re-parsing saved plan JSONs on a machine without a DB) must not require
# them. Import lazily with a sentinel so cmd_run still fails loudly if the
# DB driver is missing.
try:
    import psycopg
    import pyarrow.parquet as pq
    from pgvector.psycopg import register_vector
    _HAS_DB_DEPS = True
except ModuleNotFoundError:
    psycopg = None   # type: ignore
    pq = None        # type: ignore
    register_vector = None  # type: ignore
    _HAS_DB_DEPS = False

# Plan parser: VEC_OPS, PROPAGATION_STOPPERS, classify_node, node_wall_time,
# parse_plan, walk_plan (legacy alias). See postgres-scripts/plan_parser.py.
from plan_parser import (
    VEC_OPS,
    PROPAGATION_STOPPERS,
    classify_node,
    detect_ann_fallback,
    node_wall_time,
    parse_plan,
    parse_plan_with_residual,
    walk_plan,
)

# --- Cache flushing (try import, fall back to inline) ---
try:
    sys.path.insert(0, "/usr/local/bin")
    from flush_caches import flush_cpu_caches  # type: ignore
except ImportError:
    def flush_cpu_caches(flush_bytes: int = 256 * 1024 * 1024) -> None:
        """Fallback inline cache flush if flush_caches.py not on path."""
        import ctypes
        buf = bytearray(flush_bytes)
        ctypes.memset((ctypes.c_char * flush_bytes).from_buffer(buf), 1, flush_bytes)
        acc = 0
        mv = memoryview(buf)
        for i in range(0, flush_bytes, 64):
            acc += mv[i]
        if acc == -1:
            print(acc, file=sys.stderr)


# ============================================================================
# QuerySpec dataclass
# ============================================================================

@dataclass(frozen=True)
class QuerySpec:
    """Complete metadata for a VECH query."""
    name: str                                # "q2_start" (no "vech_" prefix)
    sql: str                                 # SQL template with {k}, %s, %(qN)s
    primary_table: str                       # "reviews" or "images"
    n_queries: int = 1                       # embeddings to load (10 for q1_start)
    positional_count: int = 0                # number of %s placeholders
    named_params: tuple = ()                 # ("q1","q2",...) for named placeholders
    needs_k: bool = True                     # SQL template has {k}
    needs_radius: bool = False               # SQL template has {radius} (ranged variants)
    multi_modal: bool = False                # second embedding from a different table
    second_table: str = ""                   # e.g. "images" for q19_start
    distinct_params: bool = False            # q1_start, q1_end
    data_driven_vs: bool = False             # q11_end: no external embeddings
    default_k: int = 100                     # q11_end overrides to 1050
    is_default: bool = True                  # included in default run set
    # Per-query measured-rep cap. None = use args.nreps. Set on very-long
    # queries (e.g. q11_end at ~12 min/rep) so a 20-rep run doesn't take 4h.
    max_nreps: Optional[int] = None


# ============================================================================
# Maximus parse_caliper schema (mirrors related_code/Maximus-VS-branch/results/parse_caliper.py:86-88)
# Used for the per_rep + aggregate CSV column ordering so plot_paper.py can consume the
# pgvector CSVs (which we write under parse_postgres/, not parse_caliper/).
# ============================================================================

REORDERED_SCHEMA = [
    # Same column layout used by both the SCALED and NO_SCALE CSV variants:
    #   - SCALED:   Total = bare_ms,  per-op columns are TIMING ON values
    #               proportionally scaled to sum to Total. OpScaleFactor =
    #               bare_ms / sum(raw operators).
    #   - NO_SCALE: Total = timed_ms, per-op columns are raw TIMING ON values
    #               so they sum to Total. OpScaleFactor = 1.0.
    # Bare and Timed columns hold bare_ms and timed_ms in both variants for
    # cross-checking. The two variants are written to parallel folders, the
    # plot script picks which one to read.
    "Total",
    "Operators", "Data Transfers", "Other", "IndexMovement",
    "Filter", "Project", "Join", "VectorSearch", "GroupBy", "OrderBy",
    "Distinct", "Limit", "LimitPerGroup", "Take", "LocalBroadcast",
    "Data Conversions", "Scatter", "Gather",
    # CTE Scan is pgvector-specific — time attributed to a WITH clause's
    # consumer scan. Rolled into "Operators" so plot 1's Rel. Operators
    # segment surfaces it; plot 2 folds it back into Other (see
    # scripts/plot_paper.py _FOLD_INTO_OTHER).
    "CTE Scan",
    "Bare", "Timed", "OpScaleFactor",
]

# Single-case label for pgvector (CPU-only). Matches Maximus's "0: CPU-CPU-CPU" format.
PGVECTOR_CASE = "0: CPU-CPU-CPU"

# Sentinel written into per_rep CSV columns for reps that postgres cancelled
# via statement_timeout. Aggregate CSV skips timed-out reps so stats stay clean.
TIMEOUT_SENTINEL = -9999.0

# Per-operator classification and time attribution live in plan_parser.py
# (small composable functions, verifiable offline via verify_plan_parser.py).
# Re-exported here so scripts importing VEC_OPS/PROPAGATION_STOPPERS from
# vech_benchmark keep working.


# ============================================================================
# Table -> column mapping (inlined from utils.py:127-147)
# ============================================================================

TABLE_INFO = {
    "reviews": {
        "id_col": "rv_reviewkey",
        "vec_col": "rv_embedding",
        "partkey_col": "rv_partkey",
    },
    "images": {
        "id_col": "i_imagekey",
        "vec_col": "i_embedding",
        "partkey_col": "i_partkey",
    },
}


# ============================================================================
# SQL Templates (verbatim from config.py:2173-2878)
# ============================================================================

SQL_Q2_START = """
SELECT
    vs.i_imagekey,
    vs.vs_distance,
    s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment
FROM
    part, supplier, partsupp, nation, region,
    (
        SELECT i_partkey, i_imagekey, i_embedding <#> %s as vs_distance
        FROM images
        WHERE i_variant = 'MAIN'
        ORDER BY i_embedding <#> %s
        LIMIT {k}
    ) vs
WHERE
    p_partkey = ps_partkey
    and s_suppkey = ps_suppkey
    and p_partkey = vs.i_partkey
    and s_nationkey = n_nationkey
    and n_regionkey = r_regionkey
    and r_name = 'EUROPE'
    and ps_supplycost = (
        select min(ps_supplycost)
        from partsupp, supplier, nation, region
        where p_partkey = ps_partkey
            and s_suppkey = ps_suppkey
            and s_nationkey = n_nationkey
            and n_regionkey = r_regionkey
            and r_name = 'EUROPE'
    )
ORDER BY
    s_acctbal DESC, vs.vs_distance ASC, n_name, s_name, p_partkey
LIMIT 100;
"""

SQL_Q18_MID = """
SELECT
    c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice,
    sum(l_quantity) as total_qty,
    sum(CASE WHEN i_partkey IS NOT NULL THEN l_quantity ELSE 0 END) as similar_qty,
    COUNT(i_partkey) as num_similar_items
FROM
    customer
    JOIN orders ON c_custkey = o_custkey
    JOIN lineitem ON o_orderkey = l_orderkey
    LEFT JOIN
        (
            SELECT i_partkey
            FROM images
            WHERE i_variant = 'MAIN'
            ORDER BY i_embedding <#> %s
            LIMIT {k}
        ) vs ON l_partkey = i_partkey
WHERE
    o_orderkey IN (
        SELECT l_orderkey FROM lineitem
        GROUP BY l_orderkey HAVING sum(l_quantity) > 300
    )
GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
ORDER BY similar_qty DESC, o_totalprice DESC, o_orderdate
LIMIT 100;
"""

SQL_Q18_MID_RANGED = """
SELECT
    c_name, o_custkey, o_orderkey, o_orderdate, o_totalprice,
    sum(l_quantity) as total_qty,
    sum(CASE WHEN i_imagekey IS NOT NULL THEN l_quantity ELSE 0 END) as similar_qty
FROM
    customer
    JOIN orders ON c_custkey = o_custkey
    JOIN lineitem ON o_orderkey = l_orderkey
    LEFT JOIN images ON l_partkey = i_partkey
        AND i_variant = 'MAIN'
        AND i_embedding <#> %s < {radius}
WHERE
    o_orderkey IN (
        SELECT l_orderkey FROM lineitem
        GROUP BY l_orderkey HAVING sum(l_quantity) > 300
    )
GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
ORDER BY similar_qty DESC, o_totalprice DESC, o_orderdate
LIMIT 100;
"""

SQL_Q10_MID = """
SELECT
    (top_k_customers.rv_custkey IS NOT NULL) as is_in_top_k,
    c_custkey, c_name,
    sum(l_extendedprice * (1 - l_discount)) as revenue,
    c_acctbal, n_name, c_address, c_phone, c_comment
FROM
    customer
    JOIN orders ON c_custkey = o_custkey
    JOIN lineitem ON o_orderkey = l_orderkey
    JOIN nation ON c_nationkey = n_nationkey
    LEFT JOIN (
        SELECT DISTINCT rv_custkey
        FROM (
            SELECT rv_custkey
            FROM reviews
            ORDER BY rv_embedding <#> %s
            LIMIT {k}
        ) AS most_similar_reviews
    ) AS top_k_customers ON c_custkey = top_k_customers.rv_custkey
WHERE
    o_orderdate >= date '1994-01-01'
    AND o_orderdate < date '1994-01-01' + interval '3' month
    AND l_returnflag = 'R'
GROUP BY
    c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment,
    top_k_customers.rv_custkey
ORDER BY is_in_top_k DESC, revenue DESC
LIMIT 20;
"""

SQL_Q16_START = """
SELECT
    p_brand, p_type, p_size,
    COUNT(DISTINCT ps_suppkey) AS supplier_cnt
FROM
    partsupp, part
WHERE
    p_partkey = ps_partkey
    AND p_brand <> 'Brand#45'
    AND p_type NOT LIKE 'MEDIUM POLISHED%%'
    AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9)
    AND ps_suppkey NOT IN (
        SELECT DISTINCT ps_suppkey
        FROM partsupp
        WHERE ps_partkey IN (
            SELECT rv_partkey
            FROM reviews
            ORDER BY rv_embedding <#> %s
            LIMIT {k}
        )
    )
GROUP BY p_brand, p_type, p_size
ORDER BY supplier_cnt DESC, p_brand, p_type, p_size;
"""

SQL_Q19_START = """
SELECT
    SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM
    lineitem, part
WHERE
    (
        p_partkey = l_partkey
        AND p_brand = 'Brand#12'
        AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
        AND l_quantity >= 1 AND l_quantity <= 11
        AND p_size BETWEEN 1 AND 5
        AND l_shipmode IN ('AIR', 'AIR REG')
        AND l_shipinstruct = 'DELIVER IN PERSON'
    )
    OR
    (
        p_partkey = l_partkey
        AND p_brand = 'Brand#23'
        AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
        AND l_quantity >= 10 AND l_quantity <= 10 + 10
        AND p_size BETWEEN 1 AND 10
        AND l_shipmode IN ('AIR', 'AIR REG')
        AND l_shipinstruct = 'DELIVER IN PERSON'
    )
    OR
    (
        p_partkey = l_partkey
        AND p_brand = 'Brand#34'
        AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
        AND l_quantity >= 20 AND l_quantity <= 30
        AND p_size BETWEEN 1 AND 15
        AND l_shipmode IN ('AIR', 'AIR REG')
        AND l_shipinstruct = 'DELIVER IN PERSON'
    )
    OR
    (
        p_partkey = l_partkey
        AND p_partkey IN (
            SELECT rv_partkey
            FROM reviews
            ORDER BY rv_embedding <#> %s
            LIMIT {k}
        )
        AND l_quantity >= 30 AND l_quantity <= 40
        AND l_shipmode IN ('AIR', 'AIR REG')
        AND l_shipinstruct = 'DELIVER IN PERSON'
    )
    OR
    (
        p_partkey = l_partkey
        AND p_partkey IN (
            SELECT i_partkey
            FROM images
            WHERE i_variant = 'MAIN'
            ORDER BY i_embedding <#> %s
            LIMIT {k}
        )
        AND l_quantity >= 40 AND l_quantity <= 50
        AND l_shipmode IN ('AIR', 'AIR REG')
        AND l_shipinstruct = 'DELIVER IN PERSON'
    );
"""

SQL_Q11_END = """
WITH TPCH_Q11_IMPORTANT_STOCK AS (
    SELECT
        ps_partkey,
        SUM(ps_supplycost * ps_availqty) AS value
    FROM
        partsupp, supplier, nation
    WHERE
        ps_suppkey = s_suppkey
        AND s_nationkey = n_nationkey
        AND n_name = 'GERMANY'
    GROUP BY ps_partkey
    HAVING
        SUM(ps_supplycost * ps_availqty) > (
            SELECT SUM(ps_supplycost * ps_availqty) * 0.0001000000
            FROM partsupp, supplier, nation
            WHERE ps_suppkey = s_suppkey
                AND s_nationkey = n_nationkey
                AND n_name = 'GERMANY'
        )
    ORDER BY value DESC
    LIMIT {k}
)
SELECT
    tpch_out.ps_partkey,
    tpch_out.value,
    dup.i_partkey AS duplicate_partkey,
    dup.dist AS visual_distance
FROM
    TPCH_Q11_IMPORTANT_STOCK tpch_out
    LEFT JOIN images query_img ON tpch_out.ps_partkey = query_img.i_partkey AND query_img.i_variant = 'MAIN'
    LEFT JOIN LATERAL (
        SELECT
            data_img.i_partkey,
            query_img.i_embedding <#> data_img.i_embedding AS dist
        FROM images data_img
        WHERE
            data_img.i_partkey != tpch_out.ps_partkey
            AND query_img.i_embedding IS NOT NULL
        ORDER BY data_img.i_embedding <#> query_img.i_embedding
        LIMIT 1
    ) dup ON TRUE
ORDER BY dup.dist, tpch_out.value DESC;
"""

SQL_Q11_END_RANGED = """
WITH TPCH_Q11_IMPORTANT_STOCK AS (
    SELECT
        ps_partkey,
        SUM(ps_supplycost * ps_availqty) AS value
    FROM partsupp, supplier, nation
    WHERE
        ps_suppkey = s_suppkey
        AND s_nationkey = n_nationkey
        AND n_name = 'GERMANY'
    GROUP BY ps_partkey
    HAVING
        SUM(ps_supplycost * ps_availqty) > (
            SELECT SUM(ps_supplycost * ps_availqty) * 0.0001000000
            FROM partsupp, supplier, nation
            WHERE ps_suppkey = s_suppkey
                AND s_nationkey = n_nationkey
                AND n_name = 'GERMANY'
        )
    ORDER BY value DESC
    LIMIT {k}
)
SELECT
    tpch_out.ps_partkey,
    tpch_out.value,
    dup.i_partkey AS duplicate_partkey,
    dup.dist AS visual_distance
FROM
    TPCH_Q11_IMPORTANT_STOCK tpch_out
    LEFT JOIN images query_img ON tpch_out.ps_partkey = query_img.i_partkey AND query_img.i_variant = 'MAIN'
    LEFT JOIN LATERAL (
        SELECT
            data_img.i_partkey,
            query_img.i_embedding <#> data_img.i_embedding AS dist
        FROM images data_img
        WHERE
            data_img.i_partkey != tpch_out.ps_partkey
            AND data_img.i_embedding <#> query_img.i_embedding < {radius}
    ) dup ON TRUE
ORDER BY dup.dist, tpch_out.value DESC;
"""

SQL_Q15_END = """
WITH revenue0 AS (
    SELECT
        l_suppkey AS supplier_no,
        SUM(l_extendedprice * (1 - l_discount)) AS total_revenue
    FROM lineitem
    WHERE
        l_shipdate >= DATE '1996-01-01'
        AND l_shipdate < DATE '1996-01-01' + INTERVAL '3' MONTH
    GROUP BY l_suppkey
),
TPCH_Q15_MAX_SUPPLIER AS (
    SELECT
        s_suppkey, s_name, s_address, s_phone, total_revenue
    FROM supplier, revenue0
    WHERE
        s_suppkey = supplier_no
        AND total_revenue = (SELECT MAX(total_revenue) FROM revenue0)
)
SELECT
    rv_reviewkey,
    tpch_q15_out.s_suppkey,
    tpch_q15_out.s_name,
    p.p_name AS part_name,
    rv_embedding <#> %s AS semantic_distance,
    rv_text
FROM
    TPCH_Q15_MAX_SUPPLIER tpch_q15_out
    JOIN partsupp ps ON tpch_q15_out.s_suppkey = ps.ps_suppkey
    JOIN part p ON ps.ps_partkey = p.p_partkey
    JOIN reviews ON p.p_partkey = rv_partkey
ORDER BY rv_embedding <#> %s
LIMIT {k};
"""

SQL_Q1_START = """
WITH review_classification AS (
    SELECT
        r.rv_partkey,
        best_match.class_name,
        best_match.distance
    FROM reviews r
    CROSS JOIN LATERAL (
        SELECT
            classes.class_name,
            (classes.class_embedding <#> r.rv_embedding) AS distance
        FROM (
            VALUES
                ('Class #0', %(q1)s),
                ('Class #1', %(q2)s),
                ('Class #2', %(q3)s),
                ('Class #3', %(q4)s),
                ('Class #4', %(q5)s),
                ('Class #5', %(q6)s),
                ('Class #6', %(q7)s),
                ('Class #7', %(q8)s),
                ('Class #8', %(q9)s),
                ('Class #9', %(q10)s)
        ) AS classes(class_name, class_embedding)
        ORDER BY distance ASC
        LIMIT 1
    ) AS best_match
),
part_winning_class AS (
    SELECT DISTINCT ON (rv_partkey)
        rv_partkey, class_name, avg_part_dist
    FROM (
        SELECT rv_partkey, class_name, COUNT(*) as frequency, AVG(distance) as avg_part_dist
        FROM review_classification
        GROUP BY rv_partkey, class_name
    ) AS counts
    ORDER BY rv_partkey, frequency DESC
)
SELECT
    l_returnflag, l_linestatus, pwc.class_name,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price,
    SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    AVG(l_quantity) AS avg_qty,
    AVG(l_extendedprice) AS avg_price,
    AVG(l_discount) AS avg_disc,
    COUNT(*) AS count_order,
    AVG(pwc.avg_part_dist) AS avg_semantic_dist
FROM
    lineitem
    LEFT JOIN part_winning_class pwc ON l_partkey = pwc.rv_partkey
WHERE l_shipdate <= DATE '1998-09-02'
GROUP BY l_returnflag, l_linestatus, pwc.class_name
ORDER BY l_returnflag, l_linestatus, pwc.class_name;
"""

SQL_Q1_END = """
SELECT
    l_returnflag, l_linestatus,
    best_match.class_name,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price,
    SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    AVG(l_quantity) AS avg_qty,
    AVG(l_extendedprice) AS avg_price,
    AVG(l_discount) AS avg_disc,
    COUNT(*) AS count_order,
    AVG(best_match.distance) AS avg_semantic_dist
FROM
    lineitem
    INNER JOIN reviews r ON l_partkey = r.rv_partkey
    CROSS JOIN LATERAL (
        SELECT
            classes.class_name,
            (classes.class_embedding <#> r.rv_embedding) AS distance
        FROM (
            VALUES
                ('Class #1', %(q1)s),
                ('Class #2', %(q2)s),
                ('Class #3', %(q3)s)
        ) AS classes(class_name, class_embedding)
        ORDER BY distance ASC
        LIMIT 1
    ) AS best_match
WHERE l_shipdate <= DATE '1998-09-02'
GROUP BY l_returnflag, l_linestatus, best_match.class_name
ORDER BY l_returnflag, l_linestatus, best_match.class_name;
"""

SQL_Q13_MID = """
SELECT
    c_count,
    COUNT(*) AS custdist,
    SUM(review_match_count) AS reviewdist
FROM
    (
        SELECT
            c_custkey,
            COUNT(DISTINCT o_orderkey) AS c_count,
            COUNT(DISTINCT top_k_reviews.rv_reviewkey) AS review_match_count
        FROM
            customer
            LEFT OUTER JOIN orders ON
                c_custkey = o_custkey
                AND o_comment NOT LIKE '%%special%%requests%%'
            LEFT OUTER JOIN (
                SELECT rv_reviewkey, rv_custkey
                FROM reviews
                ORDER BY rv_embedding <#> %s
                LIMIT {k}
            ) AS top_k_reviews ON c_custkey = top_k_reviews.rv_custkey
        GROUP BY c_custkey
    ) AS c_orders (c_custkey, c_count, review_match_count)
GROUP BY c_count
ORDER BY custdist DESC, c_count DESC;
"""

# Test/debug query: a clean mix of Filter + Join + VectorSearch with NO CTEs.
# Used to validate parser attribution. Each category should get a non-trivial,
# non-overwhelming share of the total — if one category gets ~100%, the
# classification rules are too greedy.
SQL_Q_TEST = """
SELECT
    r.rv_reviewkey,
    r.rv_partkey,
    p.p_name,
    p.p_size,
    r.rv_embedding <#> %s AS distance
FROM reviews r
INNER JOIN part p ON r.rv_partkey = p.p_partkey
WHERE p.p_size > 30
  AND r.rv_partkey < 100000
ORDER BY r.rv_embedding <#> %s
LIMIT {k};
"""


# ============================================================================
# Query Registry
# ============================================================================

QUERY_REGISTRY: dict = {
    "q1_start": QuerySpec(
        name="q1_start",
        sql=SQL_Q1_START,
        primary_table="reviews",
        n_queries=10,
        positional_count=0,
        named_params=("q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"),
        needs_k=False,
        distinct_params=True,
        is_default=False,
    ),
    "q1_end": QuerySpec(
        name="q1_end",
        sql=SQL_Q1_END,
        primary_table="reviews",
        n_queries=3,
        positional_count=0,
        named_params=("q1", "q2", "q3"),
        needs_k=False,
        distinct_params=True,
        is_default=False,
    ),
    "q2_start": QuerySpec(
        name="q2_start",
        sql=SQL_Q2_START,
        primary_table="images",
        positional_count=2,
    ),
    "q10_mid": QuerySpec(
        name="q10_mid",
        sql=SQL_Q10_MID,
        primary_table="reviews",
        positional_count=1,
    ),
    "q13_mid": QuerySpec(
        name="q13_mid",
        sql=SQL_Q13_MID,
        primary_table="reviews",
        positional_count=1,
    ),
    "q15_end": QuerySpec(
        name="q15_end",
        sql=SQL_Q15_END,
        primary_table="reviews",
        positional_count=2,
    ),
    "q16_start": QuerySpec(
        name="q16_start",
        sql=SQL_Q16_START,
        primary_table="reviews",
        positional_count=1,
    ),
    "q18_mid": QuerySpec(
        name="q18_mid",
        sql=SQL_Q18_MID,
        primary_table="images",
        positional_count=1,
    ),
    "q19_start": QuerySpec(
        name="q19_start",
        sql=SQL_Q19_START,
        primary_table="reviews",
        positional_count=2,
        multi_modal=True,
        second_table="images",
    ),
    "q11_end": QuerySpec(
        name="q11_end",
        sql=SQL_Q11_END,
        primary_table="images",
        positional_count=0,
        data_driven_vs=True,
        default_k=1050,
        # ~12 min/rep at SF1 — cap to 1 measured rep so warmup+1 = ~24 min total.
        max_nreps=1,
    ),
    # Ranged variants (not in default set)
    "q11_end_ranged": QuerySpec(
        name="q11_end_ranged",
        sql=SQL_Q11_END_RANGED,
        primary_table="images",
        positional_count=0,
        data_driven_vs=True,
        needs_radius=True,
        is_default=False,
    ),
    "q18_mid_ranged": QuerySpec(
        name="q18_mid_ranged",
        sql=SQL_Q18_MID_RANGED,
        primary_table="images",
        positional_count=1,
        needs_k=False,
        needs_radius=True,
        is_default=False,
    ),
    "q_test": QuerySpec(
        name="q_test",
        sql=SQL_Q_TEST,
        primary_table="reviews",
        positional_count=2,
        is_default=False,
    ),
}

# q11_end is intentionally last (by far the longest query, ~12 min at SF1).
DEFAULT_QUERIES = [
    "q2_start", "q10_mid", "q13_mid",
    "q15_end", "q16_start", "q18_mid", "q19_start", "q11_end",
]


# ============================================================================
# Embedding loading
# ============================================================================

def load_embeddings_from_parquet(parquet_path: str, vec_col: str, n: int) -> np.ndarray:
    """
    Load N embeddings from parquet.

    The query parquet files use a `<vec_col>_queries` column (e.g.
    `rv_embedding_queries`) holding the embedded query vectors. The corresponding
    table column is `<vec_col>` (e.g. `rv_embedding`). We try the `_queries`
    suffix first, then fall back to the bare name.
    """
    candidate = vec_col + "_queries"
    try:
        tbl = pq.read_table(parquet_path, columns=[candidate])
        col = candidate
    except Exception:
        tbl = pq.read_table(parquet_path, columns=[vec_col])
        col = vec_col
    embs = tbl[col].to_numpy(zero_copy_only=False)
    embs = np.vstack(embs)
    if embs.dtype != np.float32:
        embs = embs.astype(np.float32)
    return embs[:n]


def prepare_query_params(
    spec: QuerySpec,
    reviews_file: str,
    images_file: str,
) -> list:
    """
    Returns a list of param sets ready for cur.execute(sql, params).
    Each element is one execution's params: tuple, dict, or empty tuple.
    """
    # Mode 1: data-driven VS (q11_end) — no external embeddings needed
    if spec.data_driven_vs:
        return [()]

    # Mode 2: distinct params (q1_start, q1_end) — N embeddings into 1 dict
    if spec.distinct_params:
        primary_file = reviews_file if spec.primary_table == "reviews" else images_file
        vec_col = TABLE_INFO[spec.primary_table]["vec_col"]
        embs = load_embeddings_from_parquet(primary_file, vec_col, spec.n_queries)
        if len(embs) < spec.n_queries:
            raise ValueError(f"Need {spec.n_queries} embeddings for {spec.name}, got {len(embs)}")
        param_dict = {name: embs[i] for i, name in enumerate(spec.named_params)}
        return [param_dict]

    # Mode 3: multi-modal (q19_start) — embeddings from primary + second table
    if spec.multi_modal:
        files = {"reviews": reviews_file, "images": images_file}
        primary_emb = load_embeddings_from_parquet(
            files[spec.primary_table],
            TABLE_INFO[spec.primary_table]["vec_col"], 1,
        )[0]
        second_emb = load_embeddings_from_parquet(
            files[spec.second_table],
            TABLE_INFO[spec.second_table]["vec_col"], 1,
        )[0]
        return [(primary_emb, second_emb)]

    # Mode 4: simple — 1 embedding, duplicated for positional_count
    primary_file = reviews_file if spec.primary_table == "reviews" else images_file
    vec_col = TABLE_INFO[spec.primary_table]["vec_col"]
    embs = load_embeddings_from_parquet(primary_file, vec_col, spec.n_queries)
    return [tuple(embs[0] for _ in range(spec.positional_count)) for _ in range(spec.n_queries)]


# ============================================================================
# Query execution
# ============================================================================

def extract_metrics_from_explain_json(obj):
    """Parse EXPLAIN ANALYZE JSON. Returns (planning_ms, execution_ms, rows)."""
    if isinstance(obj, list) and obj:
        obj = obj[0]
    if not isinstance(obj, dict):
        return None, None, None
    p_time = obj.get("Planning Time")
    e_time = obj.get("Execution Time")
    rows = obj["Plan"].get("Actual Rows") if "Plan" in obj else None
    return p_time, e_time, rows


def format_sql(spec: QuerySpec, k: int, radius: float) -> str:
    """Substitute {k}, {radius}, {table}, {vec_col}, etc. into the SQL template."""
    info = TABLE_INFO[spec.primary_table]
    fmt = {
        "k": k,
        "radius": radius,
        "table": spec.primary_table,
        "vec_col": info["vec_col"],
        "id_col": info["id_col"],
        "partkey_col": info["partkey_col"],
    }
    try:
        return spec.sql.format(**fmt)
    except KeyError as e:
        raise ValueError(f"SQL template for {spec.name} requires {e}; not provided")


# ============================================================================
# Per-operator classification of EXPLAIN nodes
# ============================================================================
# classify_node, node_wall_time, walk_plan, parse_plan live in plan_parser.py
# (imported at the top of this file). Keeping the module small and verifiable
# by postgres-scripts/verify_plan_parser.py.


def dump_plan_breakdown(plan_obj, query_name: str, rep_label: str, out=sys.stderr,
                        wall_ms: Optional[float] = None,
                        scaled_operators: Optional[dict] = None,
                        bare_ms: Optional[float] = None) -> None:
    """
    Print a per-node debug dump of an EXPLAIN plan tree showing exactly what
    walk_plan() saw and how each node contributed to the operator accumulator.

    Use to diagnose sanity_check_breakdown warnings: spot which node is
    double-counted, mis-classified, or being skipped.

    Output format (one line per node):
        depth  node_type      classification    avg     loops  total   excl    rel    extras

    Where:
        avg    = node['Actual Total Time']
        loops  = node['Actual Loops']
        total  = avg * (loops / divisor)         (divisor accounts for parallel workers)
        excl   = total - sum(child totals)       (this node's exclusive contribution)
        rel    = node['Parent Relationship']     (Outer/Inner/InitPlan/SubPlan/Member)
        extras = Index Cond / Sort Key / Filter / CTE Name / Subplan Name / w=N

    InitPlan/SubPlan are flagged with [InitPlan] / [SubPlan] markers — these are
    the most common cause of double-counting because they're referenced in the
    tree multiple times.
    """
    top = plan_obj[0] if isinstance(plan_obj, list) else plan_obj
    plan_node = top.get("Plan", {})
    planning_ms = top.get("Planning Time", 0.0)
    execution_ms = top.get("Execution Time", 0.0)

    print(f"\n[debug] === {query_name} {rep_label} plan tree ===", file=out)
    print(f"[debug] Planning: {planning_ms:.2f}ms  Execution: {execution_ms:.2f}ms", file=out)
    print(f"[debug] {'depth':<5}{'node_type':<24}{'category':<13}"
          f"{'avg':>9}{'loops':>9}{'total':>10}{'excl':>10}  {'rel':<10}extras", file=out)
    print(f"[debug] {'-' * 110}", file=out)

    accum: dict = defaultdict(float)

    def _walk_for_print_only(node, depth=0, gather_loops=None, force_category=None):
        """Print an InitPlan/SubPlan subtree for visibility, but don't accumulate."""
        nt = node.get("Node Type", "?")
        avg = node.get("Actual Total Time", 0.0)
        loops = node.get("Actual Loops", 1)
        rel = node.get("Parent Relationship", "")
        is_gather = nt.startswith("Gather")
        child_gather_loops = loops if is_gather else gather_loops
        total = node_wall_time(node, gather_loops)
        category = force_category if force_category is not None else classify_node(node)
        marker = " [subplan: reattributed as VectorSearch if vec-op inside]"
        print(
            f"[debug] {depth:<5}{nt:<24}{category:<13}"
            f"{avg:>9.2f}{loops:>9}{total:>10.2f}{'':>10}  "
            f"{rel:<10}{marker}",
            file=out,
        )
        for c in node.get("Plans", []):
            _walk_for_print_only(c, depth + 1, child_gather_loops, force_category)

    def walk(node, depth=0, gather_loops=None, force_category=None, in_vec_query=False):
        nt = node.get("Node Type", "?")
        avg = node.get("Actual Total Time", 0.0)
        loops = node.get("Actual Loops", 1)
        rel = node.get("Parent Relationship", "")
        is_gather = nt.startswith("Gather")
        child_gather_loops = loops if is_gather else gather_loops

        total = node_wall_time(node, gather_loops)

        # Classification mirrors plan_parser.walk_main: stoppers reset
        # force_category, vec-Sort / vec-Index Scan enable vec-query flag,
        # bare scans of embedding tables under a vec-query become VectorSearch.
        clean_type = nt[len("Parallel "):] if nt.startswith("Parallel ") else nt
        sort_key_d = " ".join(node.get("Sort Key") or [])
        index_cond_d = node.get("Index Cond") or ""
        is_vec_sort = "Sort" in clean_type and any(op in sort_key_d for op in VEC_OPS)
        is_vec_index = any(op in index_cond_d for op in VEC_OPS)
        if is_vec_sort or is_vec_index:
            in_vec_query = True
        if clean_type in PROPAGATION_STOPPERS:
            force_category = None
        if force_category is not None:
            category = force_category
        elif in_vec_query and clean_type in ("Seq Scan", "Bitmap Heap Scan", "Index Only Scan"):
            relation = node.get("Relation Name", "")
            category = "VectorSearch" if relation in TABLE_INFO else classify_node(node)
        else:
            category = classify_node(node)
        next_force = force_category
        if category == "VectorSearch" and force_category is None:
            next_force = "VectorSearch"

        # Walk children. Skip InitPlan/SubPlan subtrees entirely to match
        # plan_parser.walk_main — their time is already folded into the parent
        # CTE Scan / referring Filter, and walk_subplans_for_vs handles VS
        # extraction from inside them separately.
        children_total = 0.0
        for c in node.get("Plans", []):
            if c.get("Parent Relationship") in ("InitPlan", "SubPlan"):
                _walk_for_print_only(c, depth + 1, child_gather_loops, next_force)
                continue
            c_total = walk(c, depth + 1, child_gather_loops, next_force, in_vec_query)
            children_total += c_total

        excl = total - children_total
        if excl < 0.0:
            # Stash the residual in Other so the debug view's running sum still
            # adds to wall time (matches plan_parser.walk_main behavior).
            accum["Other"] += excl
            excl = 0.0
        accum[category] += excl

        # Build the extras string for inspection
        extras = []
        if "Index Name" in node:
            extras.append(f"idx={node['Index Name']}")
        if "Index Cond" in node:
            extras.append(f"icond={node['Index Cond'][:35]}")
        if "Sort Key" in node:
            sk = ",".join(node["Sort Key"])[:35]
            extras.append(f"sort={sk}")
        if "Filter" in node:
            extras.append(f"filter={node['Filter'][:35]}")
        if "Subplan Name" in node:
            extras.append(f"sub={node['Subplan Name']}")
        if "CTE Name" in node:
            extras.append(f"cte={node['CTE Name']}")
        if is_gather and node.get("Workers Launched"):
            extras.append(f"w={node['Workers Launched']}")
        if rel in ("InitPlan", "SubPlan"):
            extras.insert(0, f"[{rel}]")

        marker = ""
        if rel in ("InitPlan", "SubPlan"):
            marker = " *"  # flagged: not deducted from parent
        print(
            f"[debug] {depth:<5}{nt:<24}{category:<13}"
            f"{avg:>9.2f}{loops:>9}{total:>10.2f}{excl:>10.2f}  "
            f"{rel:<10}{' '.join(extras)}{marker}",
            file=out,
        )
        return total

    walk(plan_node)

    # --- Plan stats: help diagnose why EXPLAIN ANALYZE overhead might be high ---
    # EXPLAIN ANALYZE calls clock_gettime() ~2x per row per node. So overhead
    # scales with (total rows touched) × (number of nodes). Big parallel scans
    # over many rows in deep plans get hit hard.
    stats = {"nodes": 0, "rows": 0, "max_workers": 0, "max_loops": 0}

    def _collect_stats(n):
        stats["nodes"] += 1
        rows = n.get("Actual Rows", 0) or 0
        loops = n.get("Actual Loops", 0) or 0
        stats["rows"] += rows * loops
        w = n.get("Workers Launched", 0) or 0
        if w > stats["max_workers"]:
            stats["max_workers"] = w
        if loops > stats["max_loops"]:
            stats["max_loops"] = loops
        for c in n.get("Plans", []):
            _collect_stats(c)
    _collect_stats(plan_node)

    # --- Buffer stats: how much came from shared_buffers vs OS/disk vs temp ---
    bufs = {"hit": 0, "read": 0, "dirtied": 0, "written": 0,
            "temp_read": 0, "temp_written": 0}
    def _collect_bufs(n):
        bufs["hit"] += n.get("Shared Hit Blocks", 0) or 0
        bufs["read"] += n.get("Shared Read Blocks", 0) or 0
        bufs["dirtied"] += n.get("Shared Dirtied Blocks", 0) or 0
        bufs["written"] += n.get("Shared Written Blocks", 0) or 0
        bufs["temp_read"] += n.get("Temp Read Blocks", 0) or 0
        bufs["temp_written"] += n.get("Temp Written Blocks", 0) or 0
        for c in n.get("Plans", []):
            _collect_bufs(c)
    _collect_bufs(plan_node)

    print(f"[debug] {'-' * 110}", file=out)
    print(f"[debug] Raw operator accumulator (from TIMING ON, unscaled):", file=out)
    for k, v in sorted(accum.items(), key=lambda kv: -kv[1]):
        if v > 0:
            print(f"[debug]   {k:<15} {v:>10.2f}", file=out)
    parsed_sum = sum(accum.values())
    diff_pct = (parsed_sum - execution_ms) / execution_ms * 100 if execution_ms > 0 else 0
    print(f"[debug]   {'TOTAL':<15} {parsed_sum:>10.2f}  (execution_ms TIMING ON={execution_ms:.2f}, "
          f"diff={diff_pct:+.1f}%)", file=out)

    # --- Scaled operators (proportions preserved, total matches bare execute) ---
    if scaled_operators:
        print(f"[debug]", file=out)
        print(f"[debug] Scaled operators (proportions preserved, total = bare execute):", file=out)
        for k, v in sorted(scaled_operators.items(), key=lambda kv: -kv[1]):
            if v > 0:
                print(f"[debug]   {k:<15} {v:>10.2f}", file=out)
        scaled_sum = sum(scaled_operators.values())
        print(f"[debug]   {'TOTAL':<15} {scaled_sum:>10.2f}  (this is what the CSV stores)", file=out)

    # --- Both totals side-by-side ---
    # wall_ms is the canonical scale base (currently bare_ms; previously the
    # TIMING OFF server total).
    if wall_ms is not None and wall_ms > 0:
        print(f"[debug]", file=out)
        print(f"[debug] Total comparison:", file=out)
        print(f"[debug]   bare_ms  (client-side wall):  {wall_ms:>10.2f}", file=out)
        print(f"[debug]   timed_ms (EXPLAIN TIMING ON): {execution_ms:>10.2f}  "
              f"(+{(execution_ms - wall_ms) / wall_ms * 100:.1f}% over bare)", file=out)
        if parsed_sum > 0:
            op_scale_factor = wall_ms / parsed_sum
            print(f"[debug]   op scale factor (bare/raw_ops): {op_scale_factor:>10.4f}", file=out)

    # --- Plan complexity → estimated EXPLAIN ANALYZE instrumentation cost ---
    # clock_gettime is called ~2x per row per node by PG's Instrumentation.
    # Cost per call ranges from ~27 ns (TSC, isolated, best case) to
    # ~50-100 ns under real query load due to cache pressure and bookkeeping.
    est_clock_calls = 2 * stats["rows"]
    est_low_ms = est_clock_calls * 27e-6     # best case (pg_test_timing isolated)
    est_high_ms = est_clock_calls * 100e-6   # realistic upper bound under load
    print(f"[debug]", file=out)
    print(f"[debug] Plan stats: {stats['nodes']} nodes, {stats['rows']:,} total rows "
          f"processed (sum of rows × loops), max_workers={stats['max_workers']}, "
          f"max_loops={stats['max_loops']}", file=out)
    print(f"[debug]   Estimated instrumentation cost: {est_low_ms:.0f}-{est_high_ms:.0f} ms "
          f"({est_clock_calls:,} clock_gettime() calls @ 27-100 ns each)", file=out)

    # --- Top 5 instrumentation contributors: nodes with the most row touches ---
    # These are the nodes most likely to inflate execution_ms via clock_gettime overhead.
    node_costs = []  # (node_type, rows, est_overhead_ms_low_high, extras)
    def _collect_per_node(n):
        nt = n.get("Node Type", "?")
        rows = (n.get("Actual Rows", 0) or 0) * (n.get("Actual Loops", 0) or 0)
        if rows > 0:
            extras = []
            if "Index Name" in n:
                extras.append(f"idx={n['Index Name']}")
            if "Relation Name" in n:
                extras.append(f"rel={n['Relation Name']}")
            node_costs.append((nt, rows, " ".join(extras)))
        for c in n.get("Plans", []):
            _collect_per_node(c)
    _collect_per_node(plan_node)
    node_costs.sort(key=lambda x: -x[1])
    if node_costs:
        print(f"[debug]   Top instrumentation contributors (rows × 2 × 27-100 ns):", file=out)
        for nt, rows, extras in node_costs[:5]:
            calls = 2 * rows
            low = calls * 27e-6
            high = calls * 100e-6
            print(f"[debug]     {nt:<22} {rows:>10,} rows  →  {low:>6.1f}-{high:>6.1f} ms  {extras}",
                  file=out)

    # --- Buffer summary (shared cache hits vs disk reads vs spills) ---
    # In PG, 1 block = 8 KiB. With prewarm, expect Shared Read = 0.
    PG_BLOCK_KB = 8
    total_bufs = bufs["hit"] + bufs["read"]
    hit_pct = (bufs["hit"] / total_bufs * 100) if total_bufs > 0 else 100
    print(f"[debug]", file=out)
    print(f"[debug] Buffers: shared_hit={bufs['hit']:,} blocks "
          f"({bufs['hit'] * PG_BLOCK_KB / 1024:.1f} MB)  "
          f"shared_read={bufs['read']:,} blocks "
          f"({bufs['read'] * PG_BLOCK_KB / 1024:.1f} MB from OS/disk)  "
          f"hit_ratio={hit_pct:.1f}%", file=out)
    if bufs["temp_read"] > 0 or bufs["temp_written"] > 0:
        print(f"[debug]   temp_read={bufs['temp_read']:,} blocks "
              f"({bufs['temp_read'] * PG_BLOCK_KB / 1024:.1f} MB)  "
              f"temp_written={bufs['temp_written']:,} blocks "
              f"({bufs['temp_written'] * PG_BLOCK_KB / 1024:.1f} MB)  "
              f"— query SPILLED to temp (work_mem too small for sort/hash)", file=out)
    if bufs["read"] > 0:
        print(f"[debug]   WARNING: shared_read > 0 — postgres went to OS/disk for "
              f"{bufs['read']:,} blocks. Either prewarm didn't cover this relation, "
              f"shared_buffers is too small, or pages were evicted between prewarm "
              f"and query.", file=out)

    print(f"[debug] === end {query_name} {rep_label} ===\n", file=out)


def sanity_check_breakdown(
    query_name: str,
    rep: int,
    residual_ms: float,
    execution_ms: float,
    tolerance: float = 0.05,
) -> Optional[str]:
    """
    Warn when the parser's "executor residual" — the time PG reports as
    Execution Time but does not attribute to any plan node — exceeds 5% of
    the total. This is the post-refactor descendant of the old "operator sum
    differs from execution time" warning.

    `residual_ms` is what parse_plan_with_residual() stuffed into the Other
    bucket to make sum(operators) match Execution Time exactly. A large
    residual usually means:
      - the query spilled to temp files (hash/sort > work_mem)
      - a trigger or constraint fired
      - instrumentation overhead piled up on a loop-heavy plan
    None of these are parser bugs — but they're worth surfacing because they
    distort the "per-operator time" story.
    """
    if execution_ms <= 0:
        return None
    share = abs(residual_ms) / execution_ms
    if share > tolerance:
        return (
            f"WARN: {query_name} rep={rep} {residual_ms:+.1f}ms "
            f"({share * 100:.1f}% of execution time {execution_ms:.1f}ms) "
            f"is executor residual above the plan root — likely spill, "
            f"trigger, or instrumentation overhead; time was booked to Other"
        )
    return None


def sanity_check_buffers(query_name: str, rep_label: str, plan_obj) -> Optional[str]:
    """
    Warn if the EXPLAIN (ANALYZE, BUFFERS) output shows any non-zero
    `Shared Read Blocks` — that means postgres had to go to the OS page cache
    or disk during the run. With --no_prewarm that's expected, but with prewarm
    on it indicates either (a) the table exceeds shared_buffers, or
    (b) prewarm didn't cover this relation, or (c) concurrent activity
    evicted pages between prewarm and query.

    Also warns on temp reads/writes (query spilled to disk, e.g. a big sort
    that didn't fit in work_mem).
    """
    top = plan_obj[0] if isinstance(plan_obj, list) else plan_obj
    plan_node = top.get("Plan", {})

    totals = {"shared_read": 0, "temp_read": 0, "temp_written": 0}

    def walk(n):
        totals["shared_read"] += n.get("Shared Read Blocks", 0) or 0
        totals["temp_read"] += n.get("Temp Read Blocks", 0) or 0
        totals["temp_written"] += n.get("Temp Written Blocks", 0) or 0
        for c in n.get("Plans", []):
            walk(c)
    walk(plan_node)

    issues = []
    # A page is 8 KiB in PG. Report sizes in MB so they're readable.
    if totals["shared_read"] > 0:
        mb = totals["shared_read"] * 8 / 1024
        issues.append(f"{totals['shared_read']:,} shared read blocks ({mb:.1f} MB from OS/disk)")
    if totals["temp_read"] > 0:
        mb = totals["temp_read"] * 8 / 1024
        issues.append(f"{totals['temp_read']:,} temp read blocks ({mb:.1f} MB — query spilled)")
    if totals["temp_written"] > 0:
        mb = totals["temp_written"] * 8 / 1024
        issues.append(f"{totals['temp_written']:,} temp written blocks ({mb:.1f} MB — query spilled)")

    if issues:
        return (
            f"WARN: {query_name} {rep_label} buffer sanity: {'; '.join(issues)}. "
            f"Expected 0 with --prewarm. "
            f"Check shared_buffers sizing or that prewarm covered all relations."
        )
    return None


def sanity_check_ann_fallback(
    query_name: str,
    rep_label: str,
    plan_obj,
    index_label: str,
) -> Optional[str]:
    """
    Warn when an HNSW/IVF-labelled run produced a plan that didn't actually
    use the vector index. See `plan_parser.detect_ann_fallback` for the two
    detection rules (wrong index type, brute-force pattern on reviews/images).

    Only fires for HNSW*/IVF* labels. For ENN labels, brute force is the
    intended behavior so there's nothing to warn about.

    Fix options if this fires:
      - Pass `--force_ann` to override the planner's cost decision.
      - Tune pgvector cost inputs: `effective_cache_size`, `random_page_cost`,
        re-ANALYZE, set `hnsw.ef_search` or `ivfflat.probes`.
      - Rebuild the vector index with different parameters.
    """
    label_lower = (index_label or "").lower()
    if not (label_lower.startswith("hnsw") or label_lower.startswith("ivf")):
        return None
    problem = detect_ann_fallback(plan_obj)
    if problem is None:
        return None
    return (
        f"WARN: {query_name} {rep_label}: index_label='{index_label}' but "
        f"{problem}. Pass --force_ann, tune effective_cache_size / "
        f"random_page_cost / ef_search / probes, or rebuild the vector index."
    )


def sanity_check_overhead(
    query_name: str,
    rep_label: str,
    bare_ms: float,
    instrumented_ms: float,
    tolerance: float = 0.05,
) -> Optional[str]:
    """
    Warn when TIMING ON (per-operator source) is significantly slower than the
    bare client-side total. The gap is clock_gettime instrumentation cost.
    Both scaled and no_scale CSV variants are produced from this run, so the
    warning flags both: in the scaled CSV the per-op proportions become a
    coarser approximation, and in the no_scale CSV the per-op sum (= Total)
    is inflated relative to the real wall time.

    Tolerance is relative to bare_ms.
    """
    if bare_ms <= 0 or instrumented_ms <= 0:
        return None
    overhead = (instrumented_ms - bare_ms) / bare_ms
    if overhead > tolerance:
        return (
            f"WARN: {query_name} {rep_label}: TIMING ON ({instrumented_ms:.0f}ms) "
            f"is {overhead*100:.1f}% slower than bare ({bare_ms:.0f}ms) "
            f"(>{tolerance*100:.0f}% tolerance). Affects both CSV variants: "
            f"scaled per-op times are approximate, no_scale per-op sum is inflated."
        )
    return None


@dataclass
class RunResult:
    query_name: str
    rep: int
    planning_ms: float
    # Two total measurements (see run_single_query for why):
    #   bare_ms  = client-side bare execute (Python time.time), includes socket/Python
    #              overhead but no PG instrumentation. This is the canonical "Total" used
    #              in the paper CSVs and the base operator proportions are scaled against.
    #   timed_ms = EXPLAIN (ANALYZE) TIMING ON, inflated by clock_gettime, source of
    #              per-operator timings.
    bare_ms: float
    timed_ms: float
    num_result_rows: int
    operators: dict = field(default_factory=dict)          # raw from TIMING ON
    operators_scaled: dict = field(default_factory=dict)   # proportionally scaled so the sum matches bare_ms
    explain_json: Optional[Any] = None
    result_rows: Optional[list] = None
    column_names: Optional[list] = None
    is_warmup: bool = False
    # True if postgres cancelled the query for exceeding statement_timeout.
    # cmd_run prints TIMEOUT and skips the result from all CSVs.
    timed_out: bool = False
    # Sanity-check warnings collected during the run, printed by the caller
    # AFTER the rep timing line so they don't interleave with it.
    warnings: list = field(default_factory=list)


def run_single_query(
    conn: psycopg.Connection,
    spec: QuerySpec,
    params: Any,
    sql: str,
    rep: int,
    capture_rows: bool = False,
    save_plan: bool = False,
    debug: bool = False,
    debug_rep_label: str = "",
    plan_path_hint: Optional[str] = None,
    do_flush: bool = True,
    check_buffers: bool = True,
    quick: bool = False,
    index_label: str = "",
) -> RunResult:
    """
    Run one measured execution of a query in two phases:

    1. Bare execute — client-side wall time (includes socket + Python overhead,
       zero PG instrumentation). Stored as bare_ms. This is the canonical
       "Total" used in the SCALED paper CSVs.
    2. EXPLAIN (ANALYZE) with TIMING ON — the only way to get per-node times,
       inflated by clock_gettime. Stored as timed_ms (the canonical "Total" of
       the NO_SCALE CSVs) and used as the source of per-operator times.

    Both raw (TIMING ON, sums to timed_ms) and proportionally scaled (sums to
    bare_ms) operator dicts are computed and returned. cmd_run writes them to
    parallel scaled/ and no_scale/ CSV trees so the choice can be deferred to
    plot time.

    A sanity warning fires when timed_ms is significantly slower than bare_ms
    (i.e., the per-node clock_gettime overhead is large enough that the scaled
    operator times become a coarse approximation).

    Buffer sanity check and num_rows are sourced from the TIMING ON plan
    (phase 2), since the EXPLAIN options include BUFFERS.

    A previous design also ran an EXPLAIN (ANALYZE, TIMING OFF) phase to get
    a third "clean server-side" total. It was dropped because bare_ms is a
    cleaner reference and the extra phase doubled per-rep cost. The dropped
    phase is left commented out below for reference.

    With --quick, only phase 1 runs. All other sanity checks and operator
    extraction are skipped.
    """
    # explain_clean_sql = "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON, TIMING OFF)\n" + sql  # phase 2 (dropped)
    explain_instr_sql = "EXPLAIN (ANALYZE, COSTS, VERBOSE, BUFFERS, FORMAT JSON)\n" + sql

    warnings: list[str] = []

    def _exec(sql_text):
        if isinstance(params, dict):
            cur.execute(sql_text, params)
        elif params == ():
            cur.execute(sql_text)
        else:
            cur.execute(sql_text, params)

    def _timeout_result() -> RunResult:
        """Build a sentinel RunResult for a statement_timeout cancellation.
        cmd_run prints TIMEOUT and skips this from both CSVs."""
        return RunResult(
            query_name=spec.name,
            rep=rep,
            planning_ms=0.0,
            bare_ms=0.0,
            timed_ms=0.0,
            num_result_rows=0,
            operators={},
            operators_scaled={},
            timed_out=True,
            warnings=warnings,
        )

    with conn.cursor() as cur:
        # --- Phase 1: bare execute ---
        if do_flush:
            flush_cpu_caches()
        try:
            t_start = time.time()
            _exec(sql)
            bare_ms = (time.time() - t_start) * 1000.0
        except psycopg.errors.QueryCanceled:
            return _timeout_result()
        captured_rows: Optional[list] = None
        captured_cols: Optional[list] = None
        try:
            rows = cur.fetchall()
            if capture_rows:
                captured_rows = rows
                captured_cols = [d.name for d in cur.description] if cur.description else []
        except psycopg.ProgrammingError:
            pass

        if quick:
            # Skip everything else. Return a minimal RunResult.
            return RunResult(
                query_name=spec.name,
                rep=rep,
                planning_ms=0.0,
                bare_ms=bare_ms,
                timed_ms=0.0,
                num_result_rows=len(captured_rows) if captured_rows is not None else 0,
                operators={},
                operators_scaled={},
                result_rows=captured_rows,
                column_names=captured_cols,
                warnings=warnings,
            )

        # --- (dropped) EXPLAIN (ANALYZE, TIMING OFF) ---
        # This used to be a separate phase that produced a server-side total
        # without per-row clock_gettime. It was the base for operator scaling.
        # Replaced by bare_ms. Left here commented out for reference in case
        # we ever want to re-enable it.
        # if do_flush:
        #     flush_cpu_caches()
        # _exec(explain_clean_sql)
        # clean_plan_obj = cur.fetchall()[0][0]
        # _, server_ms, _ = extract_metrics_from_explain_json(clean_plan_obj)

        # --- Phase 2: EXPLAIN (ANALYZE) TIMING ON ---
        if do_flush:
            flush_cpu_caches()
        try:
            _exec(explain_instr_sql)
            plan_obj = cur.fetchall()[0][0]
        except psycopg.errors.QueryCanceled:
            return _timeout_result()
        planning_ms, timed_ms, num_rows = extract_metrics_from_explain_json(plan_obj)

        # The TIMING ON plan also includes BUFFERS, so the buffer sanity check
        # runs against it now that phase 2 (TIMING OFF) is gone.
        if check_buffers:
            w = sanity_check_buffers(spec.name, debug_rep_label or f"rep{rep}", plan_obj)
            if w:
                warnings.append(w)

        operators, residual_ms = parse_plan_with_residual(plan_obj)
        w = sanity_check_breakdown(spec.name, rep, residual_ms, timed_ms or 0.0)
        if w:
            warnings.append(w)

        if index_label:
            w = sanity_check_ann_fallback(
                spec.name, debug_rep_label or f"rep{rep}", plan_obj, index_label,
            )
            if w:
                warnings.append(w)

    # Always compute the scaled variant alongside the raw operators. cmd_run
    # writes BOTH to parallel CSV trees so the scaled vs. no_scale decision is
    # deferred to plot time.
    raw_sum = sum(operators.values())
    if raw_sum > 0 and bare_ms and bare_ms > 0:
        op_scale_factor = bare_ms / raw_sum
        operators_scaled = {k: v * op_scale_factor for k, v in operators.items()}
    else:
        operators_scaled = dict(operators)

    w = sanity_check_overhead(
        spec.name, debug_rep_label or f"rep{rep}",
        bare_ms or 0.0, timed_ms or 0.0,
    )
    if w:
        warnings.append(w)

    if debug:
        dump_plan_breakdown(
            plan_obj, spec.name, debug_rep_label or f"rep{rep}",
            out=sys.stderr,
            wall_ms=bare_ms,
            scaled_operators=operators_scaled,
            bare_ms=bare_ms,
        )
        if plan_path_hint:
            print(f"[debug] Raw JSON saved to: {plan_path_hint}", file=sys.stderr)
            print(f"[debug] Paste into https://explain.dalibo.com/ for the canonical view.",
                  file=sys.stderr)

    return RunResult(
        query_name=spec.name,
        rep=rep,
        planning_ms=planning_ms or 0.0,
        bare_ms=bare_ms,
        timed_ms=timed_ms or 0.0,
        num_result_rows=num_rows or 0,
        operators=operators,
        operators_scaled=operators_scaled,
        explain_json=plan_obj if save_plan else None,
        result_rows=captured_rows,
        column_names=captured_cols,
        warnings=warnings,
    )


# ============================================================================
# Postgres helpers
# ============================================================================

def connect(args) -> psycopg.Connection:
    conn = psycopg.connect(
        dbname=args.db_name,
        user=args.db_user,
        password=args.db_password,
        host=args.db_host,
        port=args.db_port,
        autocommit=True,
    )
    register_vector(conn)
    return conn


def set_index_params(conn, ef_search: int, probes: int):
    with conn.cursor() as cur:
        cur.execute(f"SET hnsw.ef_search = {ef_search};")
        cur.execute(f"SET ivfflat.probes = {probes};")
        # Always strict_order — guarantees ordered results; the relaxed_order
        # and off variants can return fewer than k results when combined with
        # post-filter WHERE clauses, which breaks query semantics.
        cur.execute("SET hnsw.iterative_scan = strict_order;")


def announce_enn_session(conn) -> None:
    """
    ENN (exact nearest neighbor) runs: nothing to force.

    The ENN path guarantees brute-force vector search by simply NOT building
    any vector index on the embedding columns — with no `<->`-compatible
    index to use, the planner's only option for `ORDER BY rv_embedding <#> ?`
    is `Sort → Seq Scan(reviews)`, which is exactly the ENN semantics.

    Previous versions also did `SET enable_indexscan = off` here, but that
    GUC is session-global and affects *every* scan, not just the embedding
    tables. It killed btree lookups on `c_custkey`, `o_custkey`, `p_partkey`,
    etc., forcing the TPC-H-style joins into hash joins on wide embedding
    rows that spilled tens of GB to temp files (observed on q15_end: 48 GB
    of temp writes per rep). Removed as of 2026-04.

    This function is a no-op that exists only to print a clear marker in
    the run header so the log is self-explanatory.
    """
    print("  [ENN] No planner forcing — ENN semantics are guaranteed by "
          "the absence of any vector index (vector indexes were not built "
          "for this label).")


def force_ann_session(conn) -> None:
    """
    OPT-IN: force ANN (approximate nearest neighbor) at the session level.

    Runs `SET enable_indexscan = on; SET enable_seqscan = off;`, which makes
    the planner prefer the built HNSW/IVFFlat index for `ORDER BY <#>` even
    in cost-model edge cases (tiny k, highly selective WHERE, stale stats).

    This is no longer the default for HNSW/IVF runs because `enable_seqscan
    = off` is also session-global: it forbids Seq Scans on tiny tables like
    `nation` (25 rows) / `region` (5 rows) / small CTEs, sometimes producing
    worse plans than what the planner would pick on its own. Enable with
    `--force_ann` on the CLI when investigating a specific query whose plan
    unexpectedly bypassed the vector index.
    """
    with conn.cursor() as cur:
        cur.execute("SET enable_indexscan = on;")
        cur.execute("SET enable_seqscan = off;")
    print("  [ANN] --force_ann: SET enable_indexscan = on; SET enable_seqscan = off; "
          "(planner forced to use the built vector index)")


def prewarm(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_prewarm;")
        cur.execute("""
            SELECT pg_prewarm(c.oid)
            FROM pg_class c
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = 'public' AND c.relkind IN ('r', 'i');
        """)
        rows = cur.fetchall()
        total = sum(r[0] for r in rows)
        print(f"  Prewarmed: {total} blocks loaded")


def format_sf(sf: float) -> str:
    """
    Format scale factor for paths and filenames.
    Integer SFs (1, 10, 100) drop the trailing '.0'; fractional SFs (0.01, 0.1)
    keep their decimal representation.
    """
    return str(int(sf)) if float(sf).is_integer() else str(sf)


def db_name_for_sf(sf: float) -> str:
    """Match the existing convention from run_pg_vech.sh."""
    if sf == 1 or sf == 1.0:
        return "vech_sf1_industrial_and_scientific_plain"
    if sf == 0.01:
        return "vech_sf0_01_industrial_and_scientific_plain"
    return f"vech_sf{sf}_industrial_and_scientific_plain"


def default_dataset_paths(sf: float) -> tuple:
    """Default parquet query file paths based on scale factor."""
    home = Path.home()
    base = home / "datasets" / "amazon-23" / "final_parquet"
    sf_str = str(int(sf)) if float(sf).is_integer() else str(sf).replace(".", "_")
    reviews = base / f"industrial_and_scientific_sf{sf_str}_reviews_queries.parquet"
    images = base / f"industrial_and_scientific_sf{sf_str}_images_queries.parquet"
    return str(reviews), str(images)


# ============================================================================
# Benchmark orchestration
# ============================================================================

def _result_to_region_dict(r: RunResult, scaled: bool) -> dict:
    """
    Map a RunResult to a {region: value} dict matching REORDERED_SCHEMA.

    `scaled` selects which variant to emit:
      - True:  Total = bare_ms,  per-op = r.operators_scaled (sum to Total).
      - False: Total = timed_ms, per-op = r.operators        (sum to Total).

    For reps that postgres cancelled via statement_timeout (`r.timed_out`),
    every numeric column gets TIMEOUT_SENTINEL (-9999) so the row is obviously
    bogus to anyone reading the CSV. write_aggregate_csv skips these reps so
    they don't poison min/mean/median.

    Some columns are always 0.0 because they're Maximus-specific concepts that don't
    exist in PostgreSQL:
      - LimitPerGroup, Take, LocalBroadcast, Scatter, Gather: Maximus dataflow ops
      - Data Conversions, Data Transfers, IndexMovement: pgvector is single-machine,
        no GPU/host transfers
      - Project: PostgreSQL has no dedicated Project node (projection is implicit)

    They're kept in the dict (zeroed) so the column structure matches Maximus and
    plot_paper.py can read pgvector CSVs without code changes.
    """
    if r.timed_out:
        return {col: TIMEOUT_SENTINEL for col in REORDERED_SCHEMA}

    if scaled:
        ops = r.operators_scaled if r.operators_scaled else r.operators
        total = r.bare_ms
        raw_sum = sum(r.operators.values()) if r.operators else 0.0
        op_scale_factor = (r.bare_ms / raw_sum) if raw_sum > 0 else 1.0
    else:
        ops = r.operators
        total = r.timed_ms
        op_scale_factor = 1.0

    operators_total = sum(
        ops.get(k, 0.0)
        for k in ("Filter", "Project", "Join", "VectorSearch", "GroupBy",
                  "OrderBy", "Distinct", "Limit", "CTE Scan")
    )

    return {
        "Total": total,                   # bare_ms (scaled) or timed_ms (no_scale)
        "Bare": r.bare_ms,                # client-side bare execute (same in both variants)
        "Timed": r.timed_ms,              # raw TIMING ON total (same in both variants)
        "OpScaleFactor": op_scale_factor, # bare_ms/raw_sum (scaled) or 1.0 (no_scale)
        "Operators": operators_total,
        "Data Transfers": 0.0,           # pgvector is single-machine
        "Other": ops.get("Other", 0.0),  # Hash, Materialize, Gather, Result, etc.
        "IndexMovement": 0.0,             # pgvector indexes live in shared_buffers
        "Filter": ops.get("Filter", 0.0),
        "Project": 0.0,                   # postgres has no dedicated Project node
        "Join": ops.get("Join", 0.0),
        "VectorSearch": ops.get("VectorSearch", 0.0),
        "GroupBy": ops.get("GroupBy", 0.0),
        "OrderBy": ops.get("OrderBy", 0.0),
        "Distinct": ops.get("Distinct", 0.0),
        "Limit": ops.get("Limit", 0.0),
        "CTE Scan": ops.get("CTE Scan", 0.0),  # pg-only; see REORDERED_SCHEMA comment
        # The rest are Maximus-specific dataflow ops with no PG equivalent.
        "LimitPerGroup": 0.0,
        "Take": 0.0,
        "LocalBroadcast": 0.0,
        "Data Conversions": 0.0,
        "Scatter": 0.0,
        "Gather": 0.0,
    }


def _query_sort_key(name: str) -> tuple:
    """Sort q1, q2, q10, q11, q13... by extracting the numeric portion."""
    m = re.match(r"q(\d+)(.*)", name)
    if m:
        return (int(m.group(1)), m.group(2))
    return (999, name)


def write_per_rep_csv(path: Path, results: list, scaled: bool) -> None:
    """
    Write Maximus-compatible per_rep CSV.
    `scaled` selects the SCALED (Total=bare_ms) or NO_SCALE (Total=timed_ms) variant.
    Columns: MultiIndex[(Region, Case)] where Region in REORDERED_SCHEMA, Case = "0: CPU-CPU-CPU"
    Rows:
      - Warmup rep:     q{name}_rep0          (matches Maximus's --incl_rep0 convention)
      - Measured reps:  q{name}_rep1..rep{N}  (1-indexed, like Maximus default)
    Both are kept for inspection but only rep1..rep{N} are aggregated.
    """
    rows: dict = {}
    # Counters per query so we can renumber measured reps starting at 1.
    # Warmup is always labeled rep0; we assume at most one warmup per query.
    rep_counters: dict = defaultdict(int)

    for r in results:
        if r.is_warmup:
            row_label = f"{r.query_name}_rep0"
        else:
            rep_counters[r.query_name] += 1
            row_label = f"{r.query_name}_rep{rep_counters[r.query_name]}"
        rows[row_label] = _result_to_region_dict(r, scaled=scaled)

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.reindex(columns=REORDERED_SCHEMA)

    # Sort by query number then by rep number (rep0 first, then rep1, rep2, ...)
    def sort_key(idx_val: str) -> tuple:
        m = re.match(r"(q\d+_\w+?)_rep(\d+)", idx_val)
        if m:
            return _query_sort_key(m.group(1)) + (int(m.group(2)),)
        return (999, idx_val, 999)

    df = df.reindex(sorted(df.index, key=sort_key))

    df.columns = pd.MultiIndex.from_product([df.columns, [PGVECTOR_CASE]])
    df.columns.name = "Region"
    df.index.name = "Query_Rep"
    df.to_csv(path)


def write_aggregate_csv(path: Path, results: list, scaled: bool) -> None:
    """
    Write Maximus-compatible aggregate CSV.
    `scaled` selects the SCALED (Total=bare_ms) or NO_SCALE (Total=timed_ms) variant.
    Same column structure as per_rep, but rows are q{name}_{min,mean,median,max,std}.

    Warmup reps are excluded from aggregate stats. They live in the per_rep CSV
    for inspection but never enter min/mean/median/max/std.
    """
    by_query: dict = defaultdict(list)
    for r in results:
        if r.is_warmup or r.timed_out:
            continue
        by_query[r.query_name].append(r)

    rows: dict = {}
    for query_name, rs in by_query.items():
        if not rs:
            continue
        per_rep_df = pd.DataFrame([_result_to_region_dict(r, scaled=scaled) for r in rs])
        for stat in ("min", "mean", "median", "max", "std"):
            row_label = f"{query_name}_{stat}"
            stat_series = getattr(per_rep_df, stat)(numeric_only=True)
            stat_dict = stat_series.to_dict()
            # std() of a 1-row group is NaN; replace with 0.0 so the CSV is clean
            stat_dict = {k: (0.0 if pd.isna(v) else v) for k, v in stat_dict.items()}
            rows[row_label] = stat_dict

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.reindex(columns=REORDERED_SCHEMA)

    # Sort by query number then by stat suffix order
    stat_order = {"min": 0, "mean": 1, "median": 2, "max": 3, "std": 4}

    def sort_key(idx_val: str) -> tuple:
        m = re.match(r"(q\d+_\w+?)_(min|mean|median|max|std)", idx_val)
        if m:
            return _query_sort_key(m.group(1)) + (stat_order[m.group(2)],)
        return (999, idx_val, 999)

    df = df.reindex(sorted(df.index, key=sort_key))

    df.columns = pd.MultiIndex.from_product([df.columns, [PGVECTOR_CASE]])
    df.columns.name = "Region"
    df.index.name = "Query_Metric"
    df.to_csv(path)


def save_query_csv(path: Path, result: RunResult) -> None:
    """Save the captured query result rows to a CSV (last rep only)."""
    if result.result_rows is None:
        return
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if result.column_names:
            w.writerow(result.column_names)
        for row in result.result_rows:
            w.writerow(row)


def save_explain_json(path: Path, plan_obj: Any) -> None:
    """Save the raw EXPLAIN JSON for debugging or re-parsing."""
    with open(path, "w") as f:
        json.dump(plan_obj, f, indent=2)


def resolve_query_alias(q: str) -> str:
    """
    Accept short aliases like 'q2', 'q10', 'q11' and resolve to the full
    VECH query name ('q2_start', 'q10_mid', 'q11_end'). Full names pass through.
    Ranged variants are excluded from prefix matching — to run them you must
    type the full name (e.g. 'q11_end_ranged').
    """
    if q in QUERY_REGISTRY:
        return q
    matches = [name for name in QUERY_REGISTRY
               if name.startswith(q + "_") and not name.endswith("_ranged")]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous query alias '{q}': matches {matches}. "
                         f"Use the full name to disambiguate.")
    raise ValueError(f"Unknown query '{q}'. Use 'list' to see options.")


def cmd_run(args):
    # Resolve queries to run (accept both 'q10' and 'q10_mid' style)
    if args.query:
        try:
            queries = [resolve_query_alias(q) for q in args.query]
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        queries = DEFAULT_QUERIES

    # Resolve dataset paths (only needed for live runs — parse_only re-reads plans).
    if not args.parse_only:
        # explicit CLI flags win, otherwise fall back to the auto-derived
        # ~/datasets/amazon-23/... layout. Fail loudly if neither resolves to
        # an existing file — silent fall-through used to manifest as
        # "No results collected" much later, which is a confusing failure mode.
        auto_reviews, auto_images = default_dataset_paths(args.sf)
        reviews_file = args.reviews_queries_file or auto_reviews
        images_file = args.images_queries_file or auto_images
        for label, path in (("reviews_queries_file", reviews_file),
                            ("images_queries_file", images_file)):
            if not Path(path).exists():
                print(f"ERROR: --{label} not found at: {path}", file=sys.stderr)
                print(f"       Pass --{label} explicitly, or symlink it under "
                      f"~/datasets/amazon-23/final_parquet/.", file=sys.stderr)
                sys.exit(2)
    else:
        reviews_file = images_file = None  # unused in parse_only

    # Resolve db_name (only relevant for live runs).
    if not args.parse_only and not args.db_name:
        args.db_name = db_name_for_sf(args.sf)

    # --- Output directory layout (mirrors Maximus parse_caliper schema, but under parse_postgres/) ---
    sf_str = format_sf(args.sf)
    output_dir = Path(args.output_dir or f"./results/pg_vech_sf{sf_str}")
    # Two parallel CSV trees so the scaled-vs-no_scale decision is deferred to
    # plot time. Each holds an aggregate CSV and a per_rep/ subdir.
    #   - scaled/   Total = bare_ms,  per-op proportionally scaled to bare_ms
    #   - no_scale/ Total = timed_ms, per-op are raw TIMING ON values
    # NOTE: keep the literal "vsds" segment — plot_paper.py (in maxvec-paper/scripts/)
    # consumes parse_postgres/vsds/scaled/ by default and shares this naming with
    # Maximus's parse_caliper/vsds/ layout. Renaming would break the plotting pipeline.
    scaled_dir = output_dir / "parse_postgres" / "vsds" / "scaled"
    no_scale_dir = output_dir / "parse_postgres" / "vsds" / "no_scale"
    scaled_per_rep_dir = scaled_dir / "per_rep"
    no_scale_per_rep_dir = no_scale_dir / "per_rep"
    # Per-phase subdirs so csv/ and plans/ don't get overwritten when multiple
    # index_labels (HNSW32, IVF1024, none, ...) share the same output_dir.
    csv_dir = output_dir / "csv" / args.index_label
    plans_dir = output_dir / "plans" / args.index_label

    for d in (scaled_dir, no_scale_dir, scaled_per_rep_dir, no_scale_per_rep_dir):
        d.mkdir(parents=True, exist_ok=True)
    if args.save_csv:
        csv_dir.mkdir(parents=True, exist_ok=True)
    # --debug implies --save_plans (so the JSON is on disk and the dalibo hint
    # actually points at a real file).
    if args.debug:
        args.save_plans = True
    if args.save_plans:
        plans_dir.mkdir(parents=True, exist_ok=True)

    # --- Progress + full logs (Maximus-style) ---
    progress_path = output_dir / "progress.log"
    full_log_path = output_dir / f"vech_benchmark_{args.index_label}_sf_{sf_str}.log"

    def progress(msg: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
        with open(progress_path, "a") as f:
            f.write(line + "\n")

    # Tee class: write to both stdout and a log file simultaneously.
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                st.write(s)
                st.flush()
        def flush(self):
            for st in self.streams:
                st.flush()

    full_log = open(full_log_path, "w")
    sys.stdout = _Tee(sys.__stdout__, full_log)
    sys.stderr = _Tee(sys.__stderr__, full_log)

    # Output filename matches Maximus convention: <system>_<benchmark>_<index>_sf_<sf>_k_<k>.csv
    k_for_filename = args.k if args.k is not None else 100
    # NOTE: keep "pgvector_vsds_" prefix — consumed by plot_paper.py.
    fname = f"pgvector_vsds_{args.index_label}_sf_{sf_str}_k_{k_for_filename}.csv"
    scaled_aggregate_path = scaled_dir / fname
    scaled_per_rep_path = scaled_per_rep_dir / fname
    no_scale_aggregate_path = no_scale_dir / fname
    no_scale_per_rep_path = no_scale_per_rep_dir / fname

    print("=" * 60)
    print("VECH Benchmark Run")
    print(f"  DB:                  {args.db_host}:{args.db_port}/{args.db_name}")
    print(f"  SF:                  {args.sf}")
    print(f"  Queries:             {queries}")
    print(f"  Nreps:               {args.nreps}")
    print(f"  Index label:         {args.index_label}")
    print(f"  k:                   {args.k if args.k is not None else 'per-query default'}")
    print(f"  radius:              {args.radius}")
    print(f"  hnsw.ef_search:      {args.ef_search}")
    print(f"  ivfflat.probes:      {args.probes}")
    print(f"  hnsw.iterative_scan: strict_order")
    print(f"  Scaled aggregate:    {scaled_aggregate_path}")
    print(f"  Scaled per-rep:      {scaled_per_rep_path}")
    print(f"  No-scale aggregate:  {no_scale_aggregate_path}")
    print(f"  No-scale per-rep:    {no_scale_per_rep_path}")
    print(f"  Progress:            {progress_path}")
    print(f"  Full log:            {full_log_path}")
    print(f"  Save CSV:            {args.save_csv} ({csv_dir if args.save_csv else 'off'})")
    print(f"  Save plans:          {args.save_plans} ({plans_dir if args.save_plans else 'off'})")
    print(f"  Stmt timeout:        {args.timeout_min} min (postgres statement_timeout)")
    print(f"  Reviews:             {reviews_file}")
    print(f"  Images:              {images_file}")
    print("=" * 60)

    progress(f"START run sf={args.sf} index={args.index_label} queries={queries} nreps={args.nreps}"
             + (" [parse_only]" if args.parse_only else ""))
    overall_start = time.time()

    conn = None
    per_rep_df = None
    if args.parse_only:
        # Re-parse saved plans against the existing per_rep CSV's Bare/Timed.
        # No DB connection, no prewarm, no planner setup — just replay the plan
        # JSONs through parse_plan_with_residual and re-emit the CSVs.
        src_per_rep = scaled_per_rep_path if scaled_per_rep_path.exists() else no_scale_per_rep_path
        if not src_per_rep.exists():
            print(f"ERROR: --parse_only needs an existing per_rep CSV for Bare/Timed. "
                  f"Looked for {scaled_per_rep_path} and {no_scale_per_rep_path}.",
                  file=sys.stderr)
            sys.exit(2)
        per_rep_df = pd.read_csv(src_per_rep, header=[0, 1], index_col=0)
        print(f"  [parse_only] source per_rep: {src_per_rep}")
        print(f"  [parse_only] plans dir:      {plans_dir}")
    else:
        conn = connect(args)
        set_index_params(conn, args.ef_search, args.probes)
        # Per-statement timeout. Reps that exceed this raise psycopg.errors.QueryCanceled,
        # which run_single_query catches and reports as TIMEOUT.
        timeout_ms = args.timeout_min * 60 * 1000
        with conn.cursor() as _cur:
            _cur.execute(f"SET statement_timeout = {timeout_ms};")

        # Always print the current index state so the run is self-documenting.
        # For HNSW/IVF labels, also enforce that the right index is built (unless
        # --skip_index_check). ENN never fails here — instead we force ENN at the
        # session level below, so pre-built indexes are bypassed without dropping.
        verify_index_label_matches_db(
            conn, args.index_label, strict=not args.skip_index_check
        )

        # Planner handling by index label:
        #   - enn/none/flat  → no forcing; ENN semantics come from having dropped
        #                      the vector indexes (see announce_enn_session docstring).
        #   - hnsw* / ivf*   → trust the planner by default. The vector index exists
        #                      and pgvector's cost model picks it in practice. Use
        #                      --force_ann if a specific query needs the override.
        #   - other labels   → leave planner alone (custom/experimental runs).
        label_lower = args.index_label.lower()
        if label_lower in ("enn", "none", "flat"):
            announce_enn_session(conn)
        elif label_lower.startswith("hnsw") or label_lower.startswith("ivf"):
            if getattr(args, "force_ann", False):
                force_ann_session(conn)
            else:
                print("  [ANN] Trusting planner (pass --force_ann to override "
                      "with enable_indexscan=on / enable_seqscan=off).")

        if not args.no_prewarm:
            print("\n[prewarm]")
            progress("START prewarm")
            prewarm_start = time.time()
            prewarm(conn)
            prewarm_elapsed = time.time() - prewarm_start
            progress(f"DONE  prewarm ({prewarm_elapsed:.1f}s)")

    all_results: list = []
    n_queries = len(queries)

    for q_idx, query_name in enumerate(queries, start=1):
        spec = QUERY_REGISTRY[query_name]
        # Per-query k override (e.g. q11_end uses k=1050).
        k = args.k if args.k is not None else spec.default_k
        sql = format_sql(spec, k=k, radius=args.radius)

        print(f"\n[{q_idx}/{n_queries}] {query_name}  table={spec.primary_table}  k={k}")
        progress(f"START [{q_idx}/{n_queries}] {query_name} k={k}")
        query_start = time.time()

        if args.parse_only:
            # parse-only doesn't execute anything; param_sets is a single null
            # placeholder so the inner `for params in param_sets` still iterates once.
            param_sets = [None]
        else:
            try:
                param_sets = prepare_query_params(spec, reviews_file, images_file)
            except Exception as e:
                print(f"  ERROR loading embeddings: {e}", file=sys.stderr)
                progress(f"FAIL  [{q_idx}/{n_queries}] {query_name} (embedding load: {e})")
                continue

        # Maximus-compatible rep numbering: warmup is rep0, measured are rep1..rep{nreps}.
        # Total physical reps = (1 warmup if not disabled) + nreps.
        # Loop variable `rep` here is the 0-based physical iteration; the display
        # label maps it to Maximus's convention.
        # Cache flushing happens INSIDE run_single_query, before every execute
        # (both EXPLAIN ANALYZE and bare). No flush needed here in the rep loop.
        warmup_count = 0 if args.no_warmup else 1
        # Per-query cap (e.g. q11_end). Applies only to MEASURED reps, and
        # ONLY for the ENN baseline — with an index the query is fast enough
        # to run the full rep count. hnsw/ivf keep args.nreps as-is.
        nreps_for_query = args.nreps
        is_enn_run = args.index_label.lower() in ("enn", "none", "flat")
        if is_enn_run and spec.max_nreps is not None and spec.max_nreps < nreps_for_query:
            print(f"  [{query_name}] capping measured reps from {args.nreps} -> {spec.max_nreps} "
                  f"(spec.max_nreps; ENN run, query is too slow for the full count)")
            nreps_for_query = spec.max_nreps
        total_reps = nreps_for_query + warmup_count
        # Set when a rep hits statement_timeout. We bail out of the rep loop
        # immediately because subsequent reps would just hit the same timeout.
        timeout_hit = False
        for rep in range(total_reps):
            is_warmup = rep < warmup_count
            is_last_rep = (rep == total_reps - 1)
            capture_rows = args.save_csv and is_last_rep

            # Maximus-style label: rep0 = warmup, rep1..rep{nreps_for_query} = measured
            if is_warmup:
                label = "rep0"
            else:
                measured_idx = rep - warmup_count + 1  # 1..nreps_for_query
                label = f"rep{measured_idx}"

            for params in param_sets:
                try:
                    plan_path_hint = (
                        str(plans_dir / f"{query_name}_{label}.json")
                        if args.save_plans else None
                    )
                    if args.parse_only:
                        # Replay: build a RunResult from the saved plan JSON +
                        # Bare/Timed from the existing per_rep CSV. Missing plan
                        # files yield None and the rep is silently skipped.
                        result = _build_parse_only_result(
                            plans_dir, per_rep_df, query_name, rep, label, is_warmup
                        )
                        if result is None:
                            continue
                    else:
                        # --debug only fires for measured reps; the warmup rep0 is skipped
                        # because its numbers are dominated by cold-cache effects.
                        debug_this_rep = args.debug and not is_warmup
                        result = run_single_query(
                            conn, spec, params, sql, rep,
                            capture_rows=capture_rows,
                            save_plan=args.save_plans,
                            debug=debug_this_rep,
                            debug_rep_label=label,
                            plan_path_hint=plan_path_hint,
                            do_flush=not args.no_flush,
                            # Buffer check only makes sense if prewarm is on
                            # (otherwise shared_reads are expected).
                            check_buffers=not args.no_prewarm,
                            quick=args.quick,
                            index_label=args.index_label,
                        )
                        result.is_warmup = is_warmup
                    # Print label: warmup reps are flagged so they're obviously
                    # excluded from aggregate stats. Filenames/CSV labels still
                    # use the bare "rep0" form (set above).
                    print_label = f"(warmup) {label}" if is_warmup else label
                    if result.timed_out:
                        print(f"  {print_label} TIMEOUT after {args.timeout_min} min "
                              f"(postgres statement_timeout cancelled the query)")
                        print(f"        skipping remaining reps for {query_name} "
                              f"— next rep would also time out")
                        progress(f"TIMEOUT [{q_idx}/{n_queries}] {query_name} {label} "
                                 f"(>{args.timeout_min}min, abandoning remaining reps)")
                        # Append so the per_rep CSV records the timeout (all
                        # numeric columns become TIMEOUT_SENTINEL = -9999).
                        # Aggregate CSV skips timed_out reps so stats stay clean.
                        all_results.append(result)
                        timeout_hit = True
                        break
                    all_results.append(result)
                    if args.quick:
                        print(
                            f"  {print_label} bare={result.bare_ms:.2f}ms "
                            f"rows={result.num_result_rows} [quick mode]"
                        )
                    else:
                        ops_raw = result.operators
                        ops_scaled = result.operators_scaled or result.operators
                        # Full category breakdown so nothing is hidden. Format:
                        # "bucket=raw->scaled" using scaled rounded to 1 decimal.
                        def _cat(name: str) -> str:
                            return (
                                f"{ops_raw.get(name, 0.0):.1f}->"
                                f"{ops_scaled.get(name, 0.0):.1f}"
                            )
                        print(
                            f"  {print_label} bare={result.bare_ms:.2f}ms "
                            f"timed={result.timed_ms:.2f}ms "
                            f"planning={result.planning_ms:.2f}ms "
                            f"rows={result.num_result_rows} "
                            f"vs={_cat('VectorSearch')} "
                            f"gb={_cat('GroupBy')} "
                            f"ob={_cat('OrderBy')} "
                            f"filter={_cat('Filter')} "
                            f"join={_cat('Join')} "
                            f"lim={_cat('Limit')} "
                            f"other={_cat('Other')}"
                        )
                    # Print sanity-check warnings AFTER the rep line, indented,
                    # so they don't break up the rep0/rep1/... output stream.
                    for w in result.warnings:
                        print(f"      {w}", file=sys.stderr)

                    # Save query result rows on the last rep
                    if capture_rows and result.result_rows is not None:
                        save_query_csv(csv_dir / f"{query_name}.csv", result)

                    # Save full EXPLAIN JSON if requested (1-indexed filename).
                    # parse_only never has explain_json on the RunResult (we
                    # re-read from disk), so the guard below also short-circuits.
                    if args.save_plans and result.explain_json is not None:
                        save_explain_json(
                            plans_dir / f"{query_name}_{label}.json",
                            result.explain_json,
                        )
                except Exception as e:
                    print(f"  {label} ERROR: {e}", file=sys.stderr)
                    progress(f"FAIL  [{q_idx}/{n_queries}] {query_name} {label}: {e}")
            if timeout_hit:
                # Skip remaining reps for this query — next would also time out.
                break

        # Per-query summary: median bare_ms over measured reps. bare_ms is the
        # canonical total in both quick and full modes now that phase 2 is gone.
        measured = [r for r in all_results if r.query_name == query_name and not r.is_warmup]
        if measured:
            totals = sorted(r.bare_ms for r in measured)
            label_str = "bare"
            median_total = totals[len(totals) // 2]
            query_elapsed = time.time() - query_start
            print(f"  -> median {label_str}: {median_total:.1f} ms over {len(measured)} reps "
                  f"(query elapsed {query_elapsed:.1f}s)")
            progress(f"DONE  [{q_idx}/{n_queries}] {query_name} "
                     f"({query_elapsed:.1f}s, median {label_str}={median_total:.1f}ms, n={len(measured)})")
        else:
            progress(f"DONE  [{q_idx}/{n_queries}] {query_name} (no measured reps)")

    if conn is not None:
        conn.close()

    # --- Write the multi-index CSVs (Maximus parse_caliper format) ---
    # Two parallel trees: scaled/ (Total = bare_ms) and no_scale/ (Total = timed_ms).
    # Reps cancelled by statement_timeout appear in per_rep CSVs with all
    # numeric columns set to TIMEOUT_SENTINEL (-9999), and are skipped from
    # the aggregate CSVs so min/mean/median aren't poisoned.
    overall_elapsed = time.time() - overall_start
    n_timeouts = sum(1 for r in all_results if r.timed_out)
    if all_results:
        write_per_rep_csv(scaled_per_rep_path, all_results, scaled=True)
        write_aggregate_csv(scaled_aggregate_path, all_results, scaled=True)
        write_per_rep_csv(no_scale_per_rep_path, all_results, scaled=False)
        write_aggregate_csv(no_scale_aggregate_path, all_results, scaled=False)
        print(f"\nDone in {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min).")
        print(f"  Scaled per-rep:     {scaled_per_rep_path}")
        print(f"  Scaled aggregate:   {scaled_aggregate_path}")
        print(f"  No-scale per-rep:   {no_scale_per_rep_path}")
        print(f"  No-scale aggregate: {no_scale_aggregate_path}")
        print(f"  Timeouts:           {n_timeouts} (rows in per_rep CSVs are -9999)")
        print(f"  Progress:           {progress_path}")
        print(f"  Full log:           {full_log_path}")
        progress(f"DONE  run sf={args.sf} index={args.index_label} "
                 f"({overall_elapsed:.1f}s, timeouts={n_timeouts})")
    else:
        print("\nNo results collected.")
        progress(f"FAIL  run sf={args.sf} index={args.index_label} (no results)")

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    full_log.close()


def drop_all_vector_indexes(conn, tables: list) -> int:
    """
    Drop every HNSW and IVFFlat index on the given tables, regardless of name or params.
    Handles crash-leftover INVALID indexes too. Returns the count of indexes dropped.
    """
    indexes = query_vector_indexes(conn)
    dropped = 0
    for table, idxname, am, _ in indexes:
        if table not in tables:
            continue
        if am not in ("hnsw", "ivfflat"):
            continue
        sql = f"DROP INDEX IF EXISTS {idxname};"
        print(f"  [{table}] dropping {idxname} [{am}]")
        t = time.time()
        with conn.cursor() as cur:
            cur.execute(sql)
        print(f"    done ({time.time() - t:.1f}s)")
        dropped += 1
    return dropped


def cmd_build_indexes(args):
    if not args.db_name:
        args.db_name = db_name_for_sf(args.sf)

    tables = ["reviews", "images"] if args.table == "both" else [args.table]
    conn = connect(args)

    # --drop_vec_indexes: nuke HNSW/IVFFlat indexes on target tables, build nothing.
    # (Btree primary keys and non-vector indexes are NOT touched — see
    # drop_all_vector_indexes for the filter.)
    if args.drop_vec_indexes:
        print(f"[drop_vec_indexes] removing every HNSW/IVFFlat index on {tables}")
        n = drop_all_vector_indexes(conn, tables)
        print(f"  dropped {n} index(es)")
        conn.close()
        return

    # All other modes require --index
    if not args.index:
        print("ERROR: --index is required (unless --drop_vec_indexes is used)",
              file=sys.stderr)
        sys.exit(1)

    # --clean: nuke all existing vector indexes first, then build the requested one
    if args.clean and not args.drop:
        print(f"[clean] removing existing vector indexes on {tables} before build")
        n = drop_all_vector_indexes(conn, tables)
        print(f"  dropped {n} index(es)")

    for table in tables:
        info = TABLE_INFO[table]
        vec_col = info["vec_col"]

        # pgvector operator class must match the operator used by queries.
        # Our VECH queries use <#> (negative inner product), so the index must
        # be built with vector_ip_ops. This matches Maximus's metric=IP default.
        ops_class = {
            "ip":     "vector_ip_ops",
            "cosine": "vector_cosine_ops",
            "l2":     "vector_l2_ops",
        }[args.metric]

        if args.index == "hnsw":
            idx_name = f"{table}_{vec_col}_hnsw_{args.metric}_m{args.m}_efc{args.ef_construction}"
            if args.drop:
                sql = f"DROP INDEX IF EXISTS {idx_name};"
            else:
                sql = (
                    f"CREATE INDEX IF NOT EXISTS {idx_name} "
                    f"ON {table} USING hnsw ({vec_col} {ops_class}) "
                    f"WITH (m = {args.m}, ef_construction = {args.ef_construction});"
                )
        elif args.index == "ivfflat":
            idx_name = f"{table}_{vec_col}_ivfflat_{args.metric}_nlists{args.n_lists}"
            if args.drop:
                sql = f"DROP INDEX IF EXISTS {idx_name};"
            else:
                sql = (
                    f"CREATE INDEX IF NOT EXISTS {idx_name} "
                    f"ON {table} USING ivfflat ({vec_col} {ops_class}) "
                    f"WITH (lists = {args.n_lists});"
                )
        else:
            print(f"ERROR: unknown index type '{args.index}'", file=sys.stderr)
            sys.exit(1)

        print(f"[{table}] {sql}")
        t_start = time.time()
        with conn.cursor() as cur:
            cur.execute(sql)
        elapsed = time.time() - t_start
        action = "Dropped" if args.drop else "Built"
        print(f"  {action} in {elapsed:.1f}s")

    conn.close()


def cmd_list(args):
    print(f"{'Query':<20} {'Table':<10} {'Mode':<20} {'Default':<8}")
    print("-" * 60)
    for name, spec in QUERY_REGISTRY.items():
        if spec.distinct_params:
            mode = f"distinct ({spec.n_queries})"
        elif spec.multi_modal:
            mode = "multi-modal"
        elif spec.data_driven_vs:
            mode = "data-driven"
        else:
            mode = f"simple ({spec.positional_count}x)"
        default = "yes" if spec.is_default else "no"
        print(f"{name:<20} {spec.primary_table:<10} {mode:<20} {default:<8}")


def query_vector_indexes(conn) -> list:
    """Return [(table, indexname, am_method, indexdef), ...] for all non-btree indexes
    on reviews/images. am_method is 'hnsw', 'ivfflat', or other."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                t.relname AS tablename,
                c.relname AS indexname,
                am.amname AS access_method,
                pg_get_indexdef(c.oid) AS indexdef
            FROM pg_class c
            JOIN pg_index i ON c.oid = i.indexrelid
            JOIN pg_class t ON i.indrelid = t.oid
            JOIN pg_am am ON c.relam = am.oid
            JOIN pg_namespace n ON t.relnamespace = n.oid
            WHERE n.nspname = 'public'
              AND t.relname IN ('reviews', 'images')
              AND am.amname != 'btree';
        """)
        return list(cur.fetchall())


def verify_index_label_matches_db(conn, index_label: str, strict: bool = True) -> None:
    """
    Sanity check: confirm the indexes on reviews/images match `--index_label`,
    and always print what's physically built in the DB.

    Rules (case-insensitive prefix match on the label):
      - "enn", "none", or "flat": FAILS (in strict mode) if any vector index
                                  exists on reviews/images — the ENN path no
                                  longer forces enable_indexscan=off, so the
                                  planner will actually use a pre-built
                                  vector index if one is present, which is
                                  the opposite of what the ENN label means.
                                  Drop the index first.
      - "HNSW...":                expect at least one HNSW index, and NO IVFFlat index.
      - "IVF...":                 expect at least one IVFFlat index, and NO HNSW index.
      - anything else:            no check, just print what's there.

    Mismatches: if `strict`, sys.exit(1); else warn.
    Bypass with --skip_index_check.
    """
    indexes = query_vector_indexes(conn)
    has_hnsw = any(row[2] == "hnsw" for row in indexes)
    has_ivf = any(row[2] == "ivfflat" for row in indexes)

    actual_summary = [f"{idxname} [{am}]" for _, idxname, am, _ in indexes]
    actual_str = ", ".join(actual_summary) if actual_summary else "no vector indexes"

    # Always print the current index state so a run is self-documenting.
    print(f"  Built vector indexes on reviews/images: {actual_str}")

    label_lower = index_label.lower()

    # ENN: since we no longer set enable_indexscan=off, a pre-built vector
    # index would actually get used by the planner, defeating the ENN
    # baseline. Require absence.
    if label_lower in ("enn", "none", "flat"):
        if has_hnsw or has_ivf:
            msg = (f"Index check FAILED: --index_label='{index_label}' requires NO "
                   f"vector index on reviews/images (ENN semantics come from index "
                   f"absence, not GUC forcing), but database has: {actual_str}")
            if strict:
                print(f"  ERROR: {msg}", file=sys.stderr)
                print(f"  Drop the vector indexes first (build-indexes --drop) "
                      f"or pass --skip_index_check to bypass.", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"  WARN: {msg}", file=sys.stderr)
                return
        print(f"  Index check: OK — label='{index_label}' (no indexes, clean baseline)")
        return

    if label_lower.startswith("hnsw"):
        expected = "HNSW vector index, no IVFFlat"
        ok = (has_hnsw and not has_ivf)
    elif label_lower.startswith("ivf"):
        expected = "IVFFlat vector index, no HNSW"
        ok = (has_ivf and not has_hnsw)
    else:
        print(f"  Index check: label='{index_label}' (no rule, skipping)")
        return

    if ok:
        print(f"  Index check: OK — label='{index_label}'")
        return

    msg = (f"Index check FAILED: --index_label='{index_label}' expects "
           f"{expected}, but database has: {actual_str}")
    if strict:
        print(f"  ERROR: {msg}", file=sys.stderr)
        print(f"  Either fix the label, build the right index with `build-indexes`, "
              f"or pass --skip_index_check to bypass.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"  WARN: {msg}", file=sys.stderr)


def cmd_check_indexes(args):
    """Show all indexes on the reviews and images tables, with type and params."""
    if not args.db_name:
        if args.sf is None:
            print("ERROR: must provide either --db_name or --sf so a db_name can be derived",
                  file=sys.stderr)
            sys.exit(1)
        args.db_name = db_name_for_sf(args.sf)

    conn = connect(args)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                i.tablename,
                i.indexname,
                pg_size_pretty(pg_relation_size(quote_ident(i.indexname)::regclass)) AS index_size,
                i.indexdef
            FROM pg_indexes i
            WHERE i.schemaname = 'public'
              AND i.tablename IN ('reviews', 'images')
            ORDER BY i.tablename, i.indexname;
        """)
        rows = cur.fetchall()

    if not rows:
        print(f"No indexes found on reviews/images in {args.db_name}.")
        conn.close()
        return

    current = None
    for tablename, indexname, size, indexdef in rows:
        if tablename != current:
            current = tablename
            print(f"\n=== {tablename} ===")
        print(f"  {indexname}  ({size})")
        print(f"    {indexdef}")
    print()
    conn.close()


# ============================================================================
# CLI
# ============================================================================

def _build_parse_only_result(
    plans_dir: Path,
    per_rep_df: pd.DataFrame,
    query_name: str,
    rep: int,
    label: str,
    is_warmup: bool,
) -> Optional[RunResult]:
    """
    Build a RunResult from a saved EXPLAIN JSON + the per-rep CSV's Bare/Timed
    columns. Used by cmd_run when --parse_only is set. Returns None if the
    plan file is missing (caller skips). Timed-out rows are passed through as
    sentinel RunResults so the re-emitted CSV preserves the timeout.
    """
    row_label = f"{query_name}_{label}"
    if row_label not in per_rep_df.index:
        return None

    col = (lambda r: (r, PGVECTOR_CASE)) if isinstance(per_rep_df.columns, pd.MultiIndex) else (lambda r: r)
    bare_ms = float(per_rep_df.loc[row_label, col("Bare")])
    timed_ms = float(per_rep_df.loc[row_label, col("Timed")])

    if bare_ms == TIMEOUT_SENTINEL or timed_ms == TIMEOUT_SENTINEL:
        return RunResult(
            query_name=query_name, rep=rep, planning_ms=0.0,
            bare_ms=0.0, timed_ms=0.0, num_result_rows=0,
            operators={}, operators_scaled={},
            is_warmup=is_warmup, timed_out=True,
        )

    plan_path = plans_dir / f"{query_name}_{label}.json"
    if not plan_path.exists():
        return None

    plan_obj = json.loads(plan_path.read_text())
    operators, _residual = parse_plan_with_residual(plan_obj)

    raw_sum = sum(operators.values())
    scale = (bare_ms / raw_sum) if raw_sum > 0 else 1.0
    operators_scaled = {k: v * scale for k, v in operators.items()}

    top = plan_obj[0] if isinstance(plan_obj, list) else plan_obj
    planning_ms = float(top.get("Planning Time", 0.0) or 0.0)

    return RunResult(
        query_name=query_name, rep=rep, planning_ms=planning_ms,
        bare_ms=bare_ms, timed_ms=timed_ms, num_result_rows=0,
        operators=operators, operators_scaled=operators_scaled,
        is_warmup=is_warmup,
    )


def add_db_args(p):
    p.add_argument("--db_name", default=None, help="Auto from --sf if omitted")
    p.add_argument("--db_host", default="localhost")
    p.add_argument("--db_port", default="5432")
    p.add_argument("--db_user", default="postgres")
    p.add_argument("--db_password", default="1234")


def build_cli():
    p = argparse.ArgumentParser(description="VECH Benchmark Runner for pgvector")
    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    pr = sub.add_parser("run", help="Run VECH benchmarks")
    pr.add_argument("--sf", type=float, required=True)
    # Defaults below match Maximus's run_vech.sh:
    #   REPS=20, ivf_nprobe=30, hnsw_efsearch=128, k=100
    pr.add_argument("--nreps", type=int, default=20,
                    help="Number of measured repetitions per query (default: 20, matches Maximus)")
    pr.add_argument("--no_warmup", action="store_true",
                    help="Disable the warmup rep. By default, 1 warmup rep runs before "
                         "measurement (labeled _rep0 in the per_rep CSV, like Maximus's "
                         "Repetition_0; excluded from aggregate stats).")
    pr.add_argument("--query", nargs="+", default=None,
                    help="Specific queries (default: all default queries)")
    pr.add_argument("--k", type=int, default=None,
                    help="Override per-query default k (e.g. q11_end defaults to 1050)")
    pr.add_argument("--radius", type=float, default=0.5)
    pr.add_argument("--ef_search", type=int, default=128,
                    help="hnsw.ef_search (default: 128, matches Maximus hnsw_efsearch)")
    pr.add_argument("--probes", type=int, default=30,
                    help="ivfflat.probes (default: 30, matches Maximus ivf_nprobe)")
    pr.add_argument("--index_label", default="enn",
                    help="Index label used in output CSV filename (e.g. 'enn', 'hnsw_m16_efc64', 'ivfflat_1024')")
    pr.add_argument("--output_dir", default=None)
    pr.add_argument("--no_flush", action="store_true")
    pr.add_argument("--no_prewarm", action="store_true")
    pr.add_argument("--timeout_min", type=int, default=20,
                    help="Per-statement timeout in minutes (postgres statement_timeout). "
                         "Reps that exceed this are cancelled and reported as TIMEOUT, "
                         "and are not stored in either CSV variant. Default: 20.")
    pr.add_argument("--save_csv", action="store_true",
                    help="Save query result rows to csv/{query}.csv (last rep only). For recall comparison.")
    pr.add_argument("--save_plans", action="store_true",
                    help="Save raw EXPLAIN JSON to plans/{query}_rep{N}.json. For debugging / re-parsing.")
    pr.add_argument("--debug", action="store_true",
                    help="Dump per-node plan breakdown to stderr/log for measured reps "
                         "(rep0 warmup is skipped — its numbers are cold-cache noise). "
                         "Shows each node's classification, exclusive time, parent "
                         "relationship, wall_ms vs execution_ms comparison, and per-node "
                         "instrumentation contributors. Lines marked '*' are InitPlan/SubPlan. "
                         "Implies --save_plans and prints the JSON file path so you can "
                         "paste it into https://explain.dalibo.com/ for an independent view.")
    pr.add_argument("--skip_index_check", action="store_true",
                    help="Skip the sanity check that --index_label matches the actual indexes in the database.")
    pr.add_argument("--force_ann", action="store_true",
                    help="Force the planner to use the built HNSW/IVFFlat index by setting "
                         "enable_indexscan=on and enable_seqscan=off at the session level. "
                         "Only meaningful for hnsw*/ivf* --index_label. Default off; the planner "
                         "usually picks the vector index on its own. Use this when a specific "
                         "query's plan unexpectedly bypasses the vector index (detected by the "
                         "'brute force fallback' sanity warning in the run output).")
    pr.add_argument("--reviews_queries_file", default=None,
                    help="Path to the reviews query-embedding parquet. Overrides the auto-derived "
                         "default (~/datasets/amazon-23/final_parquet/...). Required when running "
                         "outside the original layout (e.g. on CSCS where data lives under $SCRATCH).")
    pr.add_argument("--images_queries_file", default=None,
                    help="Path to the images query-embedding parquet. Same override semantics as "
                         "--reviews_queries_file.")
    pr.add_argument("--quick", action="store_true",
                    help="Quick mode: run only the bare execute (no EXPLAIN, no per-operator "
                         "breakdown, no scaling). Fast iteration. CSV still written but with "
                         "only bare_ms populated and all operator columns empty.")
    pr.add_argument("--parse_only", action="store_true",
                    help="Skip the live benchmark. Re-parse saved EXPLAIN JSONs from "
                         "<output_dir>/plans/<index_label>/ using the current plan_parser, "
                         "reuse Bare/Timed from the existing parse_postgres/.../per_rep CSV, "
                         "and overwrite the per_rep and aggregate CSVs. Back up "
                         "parse_postgres/ yourself before running. No DB needed.")
    add_db_args(pr)
    pr.set_defaults(func=cmd_run)

    # build-indexes
    pb = sub.add_parser("build-indexes", help="Build or drop pgvector indexes")
    pb.add_argument("--sf", type=float, required=True)
    pb.add_argument("--index", choices=["hnsw", "ivfflat"], default=None,
                    help="Index type to build/drop. Required unless --drop_vec_indexes.")
    # Defaults below match Maximus's run_vech.sh:
    #   HNSW32 (m=32), IVF1024 (n_lists=1024)
    pb.add_argument("--m", type=int, default=32,
                    help="HNSW m parameter (default: 32, matches Maximus HNSW32,Flat)")
    pb.add_argument("--ef_construction", type=int, default=64,
                    help="HNSW ef_construction (default: 64)")
    pb.add_argument("--n_lists", type=int, default=1024,
                    help="IVFFlat number of lists (default: 1024, matches Maximus IVF1024,Flat)")
    pb.add_argument("--metric", choices=["ip", "cosine", "l2"], default="ip",
                    help="Distance metric / pgvector ops class (default: ip = vector_ip_ops, "
                         "matches Maximus metric=IP. Must agree with the operator the queries "
                         "use: <#> for ip, <=> for cosine, <-> for l2.)")
    pb.add_argument("--table", choices=["reviews", "images", "both"], default="both")
    pb.add_argument("--drop", action="store_true",
                    help="Drop the specific index matching --index/--m/--n_lists/--metric (by name).")
    pb.add_argument("--clean", action="store_true",
                    help="Drop ALL existing HNSW and IVFFlat indexes on target tables BEFORE building. "
                         "Idempotent: switching index types or recovering from a crash always lands "
                         "in a known state.")
    pb.add_argument("--drop_vec_indexes", "--drop_all", dest="drop_vec_indexes",
                    action="store_true",
                    help="Drop every HNSW and IVFFlat index on reviews/images "
                         "(nothing else — btree primary keys and non-vector "
                         "indexes are left alone). Build nothing. Use this to "
                         "return to the ENN (no-index) baseline. Old alias "
                         "`--drop_all` still works for back-compat.")
    add_db_args(pb)
    pb.set_defaults(func=cmd_build_indexes)

    # list
    pl = sub.add_parser("list", help="List available queries")
    pl.set_defaults(func=cmd_list)

    # check-indexes
    pc = sub.add_parser("check-indexes",
                        help="Show all indexes currently built on reviews/images")
    pc.add_argument("--sf", type=float, default=None,
                    help="Used to auto-derive db_name if --db_name not given")
    add_db_args(pc)
    pc.set_defaults(func=cmd_check_indexes)

    return p


def main():
    parser = build_cli()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
