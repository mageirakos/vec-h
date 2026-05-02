SELECT
    c_name,
    c_custkey,
    o_orderkey,
    o_orderdate,
    o_totalprice,
    sum(l_quantity) as total_qty,
    -- [+] Calculate the "Visual Score" (Only count "similar" items that matched the query image)
    sum(CASE WHEN i_partkey IS NOT NULL THEN l_quantity ELSE 0 END) as similar_qty
FROM
    customer
    JOIN orders ON c_custkey = o_custkey
    JOIN lineitem ON o_orderkey = l_orderkey
    -- [+] Attach the Vector Search (Must be LEFT JOIN to preserve non-matching orders)
    LEFT JOIN 
        (
            SELECT i_partkey
            FROM images
            WHERE i_variant = 'MAIN' -- [+] only main images --> unique partkeys
            ORDER BY i_embedding <#> %s
            LIMIT {k}
        ) vs ON l_partkey = i_partkey 
WHERE
    o_orderkey IN (
        SELECT
            l_orderkey
        FROM
            lineitem
        GROUP BY
            l_orderkey 
        HAVING
            sum(l_quantity) > 300
    )
GROUP BY
    c_name,
    c_custkey,
    o_orderkey,
    o_orderdate,
    o_totalprice
ORDER BY
    -- [+] Re-rank: Validated visual matches come first
    similar_qty DESC, 
    o_totalprice DESC,
    o_orderdate
LIMIT 100;