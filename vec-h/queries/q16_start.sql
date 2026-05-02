SELECT
	p_brand,
	p_type,
	p_size,
	COUNT(DISTINCT ps_suppkey) AS supplier_cnt
FROM
	partsupp,
	part
WHERE
	p_partkey = ps_partkey
	AND p_brand <> 'Brand#45'
	AND p_type NOT LIKE 'MEDIUM POLISHED%%'
	AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9)
	AND ps_suppkey NOT IN (
        -- [+] Identify suppliers of parts with "complaint-like" reviews
        SELECT  DISTINCT ps_suppkey
        FROM  partsupp
        WHERE ps_partkey IN (
                -- 1. Find parts associated with the most semantically similar reviews
                SELECT rv_partkey
                FROM reviews
                ORDER BY rv_embedding <#> %s
                LIMIT {k}
            )
        -- [-] remove "Supplier with comments like '%Customer%Complaints%'"
        -- SELECT s_suppkey FROM supplier WHERE s_comment LIKE '%Customer%Complaints%'
	)
GROUP BY
	p_brand,
	p_type,
	p_size
ORDER BY
	supplier_cnt DESC,
	p_brand,
	p_type,
	p_size;

