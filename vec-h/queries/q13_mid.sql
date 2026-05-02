SELECT
    c_count,
    COUNT(*) AS custdist,
    -- [+] Aggregate the total number of top-K reviews for this order-count bucket
    -- review distribution that match <review_input_vector> by number of reviews
    SUM(review_match_count) AS reviewdist
FROM
    (
        SELECT
            c_custkey,
            COUNT(DISTINCT o_orderkey) AS c_count,
            -- [+] Count how many top-K reviews belong to this customer
            COUNT(DISTINCT top_k_reviews.rv_reviewkey) AS review_match_count
        FROM
            customer 
            LEFT OUTER JOIN orders ON
                c_custkey = o_custkey
                AND o_comment NOT LIKE '%%special%%requests%%'
            
            -- [+] The Independent Vector Search Branch
            -- Finds global Top-K reviews similar to the input vector
            LEFT OUTER JOIN (
                SELECT rv_reviewkey, rv_custkey
                FROM reviews
                ORDER BY rv_embedding <#> %s
                LIMIT {k}
            ) AS top_k_reviews ON c_custkey = top_k_reviews.rv_custkey
        GROUP BY
            c_custkey
    ) AS c_orders (c_custkey, c_count, review_match_count)
GROUP BY
    c_count
ORDER BY
    custdist DESC,
    c_count DESC;