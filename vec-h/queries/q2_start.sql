SELECT
	vs.i_imagekey, -- [+] add key from retrieved images
	vs.vs_distance, -- [+] add distance
	s_acctbal,
	s_name,
	n_name,
	p_partkey,
	p_mfgr,
	s_address,
	s_phone,
	s_comment
FROM
	part,
	supplier,
	partsupp,
	nation,
	region,
	-- #### [+] add semantic filter 
	(
		SELECT i_partkey, i_imagekey, i_embedding <#> %s as dist
		FROM images
		WHERE i_variant = 'MAIN' -- [+] only main images --> unique partkeys
		ORDER BY i_embedding <#> %s
		LIMIT {k}
	) vs
	-- ####
WHERE
	p_partkey = ps_partkey
	and s_suppkey = ps_suppkey
	and p_partkey = vs.i_partkey
	-- and p_size = 38 -- [-] remove "size" 
	-- and p_type like '%TIN' -- [-] remove "type" 
	and s_nationkey = n_nationkey
	and n_regionkey = r_regionkey
	and r_name = 'EUROPE'
	and ps_supplycost = (
		select
			min(ps_supplycost)
		from
			partsupp,
			supplier,
			nation,
			region
		where
			p_partkey = ps_partkey
			and s_suppkey = ps_suppkey
			and s_nationkey = n_nationkey
			and n_regionkey = r_regionkey
			and r_name = 'EUROPE'
	)	
ORDER BY
	s_acctbal DESC,
	vs.vs_distance ASC, -- [+] secondary sort by distance (semantic similarity)
	n_name,
	s_name,
	p_partkey
LIMIT 100;
