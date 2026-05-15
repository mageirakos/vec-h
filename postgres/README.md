# Vec-H pgvector benchmark

## Run on any host (need Docker)

```bash
uv sync
source .venv/bin/activate

cd postgres-default/
# 1. Build image + start postgres container (dataset-agnostic, once)
make setup
# 2. Load a dataset into a DB (defaults: vech-industrial_and_scientific-sf_1 → DB "vech")
make load-data DATASET=vech-industrial_and_scientific-sf_1 DB_NAME=vech
# 3. Run full HNSW + IVF + ENN sweep against it
DATASET=vech-industrial_and_scientific-sf_1 DB_NAME=vech ./run_pg_vech.sh
```

To use a different dataset/DB, see [Selecting a dataset / database](#selecting-a-dataset--database).

Results: `postgres-default/results/runs/<DATE>-default/run<N>/`.

## Run on CSCS GH200 (HPC cluster w/ podman)

```bash
cd postgres-cscs/
# 1. Build image locally + import as enroot squashfs
make save-image
# 2. Transfer image to daint
./build_and_deploy.sh
# 3. initdb + load data (one-time per dataset)
sbatch setup_pgvector_cscs.sh 1
# 4. Run full HNSW + IVF + ENN sweep
sbatch run_cscs_pg_vech.sh 1
```

Quick test: `sbatch quick_test_cscs.sh`.

CSCS uses the **same `DATASET` / `DB_NAME` knobs** as the default (so a
dataset dir is laid out identically), but the base is `$SCRATCH/datasets/`
instead of repo `data/` — i.e. the data must be rsynced to
`$SCRATCH/datasets/<DATASET>/`. Override via env:

```bash
DATASET=vech-sf1 DB_NAME=vech_sf1 sbatch setup_pgvector_cscs.sh 1
DATASET=vech-sf1 DB_NAME=vech_sf1 sbatch run_cscs_pg_vech.sh 1
```

## Layout

```
postgres-scripts/    # Docker image source + benchmark runner + plan parser
postgres-default/    # docker-compose orchestration (default Docker host)
postgres-cscs/       # SLURM + Enroot orchestration (CSCS GH200)
load_to_postgres_database.py
```

## Expected data location (Docker host)

Each dataset is **one directory** under repo-root `data/`, which
`postgres-default/docker-compose.yml` bind-mounts to `/datasets` in the
container. A dataset directory looks like:

```
data/<DATASET>/
  {region,nation,part,supplier,partsupp,customer,orders,lineitem}.csv   # flat TPC-H CSVs
  reviews.parquet  images.parquet                                       # flat VECH parquet
  reviews_queries.parquet  images_queries.parquet
  tpch_parquet-sf1/                                                     # optional, unused by the loader
```

Generate one via `vec-h/dataset-generation/` (step 6 writes this layout), or
download the prebuilt SF1 set (step 0 there). Two datasets ship as examples:
`vech-industrial_and_scientific-sf_1` (default) and `vech-sf1`.

## Selecting a dataset / database

Args:

| Variable | Default | Meaning |
|---|---|---|
| `DATASET` | `vech-industrial_and_scientific-sf_1` | dir under `data/` → `/datasets/<DATASET>` |
| `DB_NAME` | `vech` | target Postgres DB (auto-created if missing) |
| `DROP_DB` | `0` | `1` → drop & recreate `DB_NAME` before loading (clean reload) |

Example with different dataset and dbname, not the industrial-scientific, you don't have to rebuild image:
```bash
make load-data DATASET=vech-sf1 DB_NAME=vech_sf1
DB_NAME=vech_sf1 DATASET=vech-sf1 ./run_pg_vech.sh 1
make psql DB_NAME=vech_sf1
```

`make load-data` appends into existing tables; So you need to drop db if an existing one is there to start fresh:

```bash
make load-data DROP_DB=1                                  # reload default DB
make load-data DROP_DB=1 DATASET=vech-sf1 DB_NAME=vech_sf1 # reload a named DB
```