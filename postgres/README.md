# Vec-H pgvector benchmark

## Expected data location

### Default (Docker host)

```
~/datasets/
  tpch-datasets/csv/csv-<SF>/{region,nation,part,supplier,partsupp,customer,orders,lineitem}.csv
  industrial_and_scientific_sf<SF>_reviews.parquet
  industrial_and_scientific_sf<SF>_images.parquet
  industrial_and_scientific_sf<SF>_reviews_queries.parquet
  industrial_and_scientific_sf<SF>_images_queries.parquet
```

### CSCS GH200

```
$SCRATCH/datasets/sf<SF>/
  tpch/{region,nation,part,supplier,partsupp,customer,orders,lineitem}.csv
  industrial_and_scientific_sf<SF>_reviews.parquet
  industrial_and_scientific_sf<SF>_images.parquet
  industrial_and_scientific_sf<SF>_reviews_queries.parquet
  industrial_and_scientific_sf<SF>_images_queries.parquet
```

## Run on default (Docker host)

```bash
cd postgres-default/
# 1. Build image + start postgres container
make setup
# 2. Load TPC-H + reviews/images (one-time per SF)
make load-data SF=1
# 3. Run full HNSW + IVF + ENN sweep
./run_pg_vech.sh 1
```

Results: `postgres-default/results/runs/<DATE>-default/run<N>/`.

## Run on CSCS GH200

```bash
cd postgres-cscs/
# 1. Build image locally + import as enroot squashfs
make save-image
# 2. Transfer image to daint
./build_and_deploy.sh
# 3. initdb + load data (one-time per SF)
sbatch setup_pgvector_cscs.sh 1
# 4. Run full HNSW + IVF + ENN sweep
sbatch run_cscs_pg_vech.sh 1
```

Quick smoke test: `sbatch quick_test_cscs.sh`.

## Layout

```
postgres-scripts/    # Docker image source + benchmark runner + plan parser
postgres-default/    # docker-compose orchestration (default Docker host)
postgres-cscs/       # SLURM + Enroot orchestration (CSCS GH200)
load_to_postgres_database.py
```
