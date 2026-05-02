# Vec-H

Artifact repository for : "To GPU or Not to GPU: Vecor Search in Relational Engines" under submission for VLDB '27

# Reproduce
If you run into any problems contact: vmageirakos@inf.ethz.ch or open an issue
## Dataset (Vec-H)
- What we used in paper: https://polybox.ethz.ch/index.php/s/tcFK2AHaMaBKHyw
- Queries found under: `vec-h/queries/*`
- To generate a new version of the dataset from scratch follow: `vec-h/dataset-generation/README.md`

## Postgres

``` bash
cd postgres/postgres-default/

# 1. Build image + start postgres container
make setup
# 2. Load data
make load-data
# 3. Run full benchmark (HNSW + IVF + ENN sweep)
./run_pg_vech.sh
```

For more details: `./postgres/README.md`


## MaxVec

``` bash
cd maxvec/

# 1. Build image + setup (apply needed patches for non-data-owning indexes)
make setup APPLY_RAFT_ATS_PATCH=true APPLY_FAISS_CAGRA_PATCH=true APPLY_FAISS_IVF_PATCH=true

# 2. Run full benchmark
make full-run SYSTEM=sgs-gpu06 STAGES=vsds,varbatch
```

For more details: `./maxvec/README.md`.  

Note: `vsds` is what we used call the now `vec-h`, just in case you see it in the code.

# Credit 
For this work: Vasilis Mageirakos, Joel André, Mako Kabić, Bowen Wu, Yannis Chornis, Gustavo Alonso (Systems Group ETH Zürich : https://systems.ethz.ch/ )

Prior code/data/work we used or extended:
- Maximus : https://gitlab.inf.ethz.ch/PUB-SYSTEMS/eth-dataprocessing/Maximus
- TPC-H dataset : https://www.tpc.org/tpch/
- DuckDB tpch-h extension: https://duckdb.org/docs/current/core_extensions/tpch 
- Amazon Reviews dataset: https://amazon-reviews-2023.github.io/
