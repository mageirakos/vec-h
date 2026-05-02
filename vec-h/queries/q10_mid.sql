SELECT
    -- [+] The Correlation Flag
    -- Checks if the customer exists in the independent "Top-K" subquery (True/False)
    (top_k_customers.rv_custkey IS NOT NULL) as is_in_top_k,
    c_custkey,
    c_name,
    sum(l_extendedprice * (1 - l_discount)) as revenue,
    c_acctbal,
    n_name,
    c_address,
    c_phone,
    c_comment
    
FROM
    customer
    -- [+] Converted to Explicit Joins (Required to use LEFT JOIN below)
    JOIN orders ON c_custkey = o_custkey
    JOIN lineitem ON o_orderkey = l_orderkey
    JOIN nation ON c_nationkey = n_nationkey
    -- [+] The Independent Vector Branch (VS @ Middle/Independent)
    -- Global Top-K customers with most similar reviews to the input review (e.g. a specific complaint)
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
    c_custkey,
    c_name,
    c_acctbal,
    c_phone,
    n_name,
    c_address,
    c_comment,
    top_k_customers.rv_custkey -- [+] Added to Group By
ORDER BY
    -- [+] Re-rank: Customer is in the Top-K subquery
    is_in_top_k ASC,
    revenue DESC
LIMIT 20;