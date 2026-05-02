WITH TPCH_Q11_IMPORTANT_STOCK AS (
    -- [Original TPCH-Q11 Query]
    SELECT
        ps_partkey,
        SUM(ps_supplycost * ps_availqty) AS value
    FROM
        partsupp,
        supplier,
        nation
    WHERE
        ps_suppkey = s_suppkey
        AND s_nationkey = n_nationkey
        AND n_name = 'GERMANY'
    GROUP BY
        ps_partkey
    HAVING
        SUM(ps_supplycost * ps_availqty) > (
            SELECT
                SUM(ps_supplycost * ps_availqty) * 0.0001000000
            FROM
                partsupp,
                supplier,
                nation
            WHERE
                ps_suppkey = s_suppkey
                AND s_nationkey = n_nationkey
                AND n_name = 'GERMANY'
        )
    ORDER BY
        value DESC
    -- [+] Adding an optional limit clause s.t. we control the "queries" in the VS
    LIMIT {k}
)
SELECT
    tpch_out.ps_partkey,
    tpch_out.value,
    -- [+] Select duplicate info
    dup.i_partkey AS duplicate_partkey,
    dup.dist AS visual_distance
FROM
    TPCH_Q11_IMPORTANT_STOCK tpch_out
    -- [+] Get the embedding for the source part
    -- note: we only search based on main image, althoug we could've search for all and keep most similar
    LEFT JOIN images query_img ON tpch_out.ps_partkey = query_img.i_partkey AND query_img.i_variant = 'MAIN'
    -- [+] Scan for the closest match for *this specific row*
    LEFT JOIN LATERAL (
        SELECT 
            data_img.i_partkey,
            query_img.i_embedding <#> data_img.i_embedding AS dist
        FROM 
            images data_img
        WHERE 
            data_img.i_partkey != tpch_out.ps_partkey -- [+] Exclude itself
            AND query_img.i_embedding IS NOT NULL -- [+] avoid NULL
        ORDER BY 
            data_img.i_embedding <#> query_img.i_embedding
        LIMIT 1 -- [+] Find top 1 duplicate (most similar )
    ) dup ON TRUE
ORDER BY
    dup.dist,
    tpch_out.value DESC;


-- Alternative "Ranged Version":
---- Ranged Search version , but no range support in FAISS on GPU for indexes, we use top-k version in paper for result consistency

WITH TPCH_Q11_IMPORTANT_STOCK AS (
    -- [Original Query Logic preserved exactly]
    SELECT
        ps_partkey,
        SUM(ps_supplycost * ps_availqty) AS value
    FROM
        partsupp,
        supplier,
        nation
    WHERE
        ps_suppkey = s_suppkey
        AND s_nationkey = n_nationkey
        AND n_name = 'GERMANY'
    GROUP BY
        ps_partkey
    HAVING
        SUM(ps_supplycost * ps_availqty) > (
            SELECT
                SUM(ps_supplycost * ps_availqty) * 0.0001000000
            FROM
                partsupp,
                supplier,
                nation
            WHERE
                ps_suppkey = s_suppkey
                AND s_nationkey = n_nationkey
                AND n_name = 'GERMANY'
        )
    ORDER BY
        value DESC
    -- [+] Adding an optional limit clause s.t. we control the "queries" in the VS
    LIMIT {k}
)
SELECT
    tpch_out.ps_partkey,
    tpch_out.value,
    -- [+] Select duplicate info
    dup.i_partkey AS duplicate_partkey,
    dup.dist AS visual_distance
FROM
    TPCH_Q11_IMPORTANT_STOCK tpch_out
    -- [+] Get the embedding for the source part
    LEFT JOIN images query_img ON tpch_out.ps_partkey = query_img.i_partkey AND query_img.i_variant = 'MAIN'
    -- [+] Scan for all matches within range for *this specific row*
    LEFT JOIN LATERAL (
        SELECT 
            data_img.i_partkey,
            data_img.i_embedding <#> query_img.i_embedding AS dist
        FROM 
            images data_img
        WHERE 
            data_img.i_partkey != tpch_out.ps_partkey -- [-] Exclude itself
            AND data_img.i_embedding <#> query_img.i_embedding < {radius} -- [+] Range filter
    ) dup ON TRUE
ORDER BY
    dup.dist, tpch_out.value DESC;