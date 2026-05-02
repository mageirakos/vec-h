WITH revenue0 AS (
    SELECT
        l_suppkey AS supplier_no,
        SUM(l_extendedprice * (1 - l_discount)) AS total_revenue
    FROM
        lineitem
    WHERE
        l_shipdate >= DATE '1996-01-01'
        AND l_shipdate < DATE '1996-01-01' + INTERVAL '3' MONTH
    GROUP BY
        l_suppkey
),
TPCH_Q15_MAX_SUPPLIER AS (
    SELECT
        s_suppkey,
        s_name,
        s_address,
        s_phone,
        total_revenue
    FROM
        supplier,
        revenue0
    WHERE
        s_suppkey = supplier_no
        AND total_revenue = (
            SELECT MAX(total_revenue) FROM revenue0
        )
)
-- [+] Find most similar reviews from PARTS of the top supplier to sume <review query> (e.g. complaint, item feature, possitive feedback etc.)
SELECT
    rv_reviewkey,
    tpch_q15_out.s_suppkey,
    tpch_q15_out.s_name,
    p.p_name AS part_name,
    rv_embedding <#> %s AS semantic_distance,
    rv_text
FROM
    TPCH_Q15_MAX_SUPPLIER tpch_q15_out
    -- [+] Join to find all parts provided by this top supplier
    JOIN partsupp ps ON tpch_q15_out.s_suppkey = ps.ps_suppkey
    JOIN part p ON ps.ps_partkey = p.p_partkey
    -- [+] Join to reviews for those specific parts
    JOIN reviews ON p.p_partkey = rv_partkey
ORDER BY
    -- [+] Sort by semantic similarity to e.g. some "feature/complaint/positive thing etc."
    rv_embedding <#> %s
LIMIT {k};