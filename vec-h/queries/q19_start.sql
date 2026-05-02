SELECT
	SUM(l_extendedprice* (1 - l_discount)) AS revenue
FROM
	lineitem,
	part
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
	-- [+] Add Reviews based Vector Search Filter
	-- "if it matches this brand description" AND is sent by AIR etc.
	-- Replaces "Brand/Container" definition with "Review Similarity"
	OR 
	(
		p_partkey = l_partkey
        AND p_partkey IN (
            SELECT rv_partkey 
            FROM reviews 
            ORDER BY rv_embedding <#> %s 
            LIMIT {k}
        )
        -- Operational constraints maintained (Air + Person), with a new quantity tier
        AND l_quantity >= 30 AND l_quantity <= 40
        AND l_shipmode IN ('AIR', 'AIR REG')
        AND l_shipinstruct = 'DELIVER IN PERSON'
	)
	-- [+] Add Images based Vector Search Filter
	-- "if it matches this visual context" AND is sent by AIR etc.
	-- Replaces "Brand/Container" definition with "Visual Similarity"
	OR
	(
		p_partkey = l_partkey
        AND p_partkey IN (
            SELECT i_partkey 
            FROM images 
            WHERE i_variant = 'MAIN' -- Optional: ensure we match main product photos
            ORDER BY i_embedding <#> %s 
            LIMIT {k}
        )
        -- Operational constraints maintained (Air + Person), with a new quantity tier
        AND l_quantity >= 40 AND l_quantity <= 50
        AND l_shipmode IN ('AIR', 'AIR REG')
        AND l_shipinstruct = 'DELIVER IN PERSON'
	);