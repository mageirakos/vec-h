ROOTLESS := $(shell docker info --format '{{json .SecurityOptions}}' | grep -q rootless && echo 1 || echo 0)
IS_ROOTLESS := $(if $(filter 1,$(ROOTLESS)),true,false)

SF ?= 1

# before loading, for a clean reload. Default 0 (append into existing tables).
DROP_DB ?= 0
DROP_DB_FLAG := $(if $(filter-out 0,$(DROP_DB)),--drop_db,)

# DATASET = directory name under repo-root data/ (bind-mounted to /datasets).
# DB_NAME = target Postgres database (auto-created by the loader if missing).
# Switch datasets/DBs at runtime, e.g.:
#   make load-data DATASET=vech-sf1 DB_NAME=vech_sf1
DATASET ?= vech-industrial_and_scientific-sf_1
DB_NAME ?= vech

# Container-side paths (the loader runs inside the container).
DATA_DIR ?= /datasets/$(DATASET)
TPCH_DIR ?= $(DATA_DIR)
REVIEWS_FILE ?= $(DATA_DIR)/reviews.parquet
IMAGES_FILE ?= $(DATA_DIR)/images.parquet
REVIEWS_Q_FILE ?= $(DATA_DIR)/reviews_queries.parquet
IMAGES_Q_FILE ?= $(DATA_DIR)/images_queries.parquet

.PHONY: setup down shell psql load-data help is-rootless

setup:
	mkdir -p ~/postgres-scratch/pg_data
	docker compose up -d --build
	@echo ""
	@echo "Postgres starting. Check: docker compose logs -f"
	@echo "Next: make load-data DATASET=$(DATASET) DB_NAME=$(DB_NAME)"

down:
	docker compose down

shell:
	docker compose exec pgvector bash

psql:
	docker compose exec pgvector psql -U postgres -d $(DB_NAME)

load-data:
	docker compose exec pgvector python3 /maxvec/load_to_postgres_database.py \
		--sf $(SF) \
		--db_name $(DB_NAME) \
		--db_host localhost \
		--db_user postgres \
		--db_password 1234 \
		$(DROP_DB_FLAG) \
		--tpch_dir $(TPCH_DIR) \
		--reviews_file $(REVIEWS_FILE) \
		--images_file $(IMAGES_FILE) \
		--reviews_queries_file $(REVIEWS_Q_FILE) \
		--images_queries_file $(IMAGES_Q_FILE)
	@echo ""
	@echo "Running ANALYZE..."
	docker compose exec pgvector psql -U postgres -d $(DB_NAME) -c "ANALYZE;"
	@echo "Done. Verify: make psql"

is-rootless:
	@echo "Docker rootless: $(IS_ROOTLESS)"

help:
	@echo "Targets:"
	@echo "  setup       - Build image + start postgres (creates ~/postgres-scratch/)"
	@echo "  load-data   - Load TPC-H + VECH data into DB_NAME from DATASET"
	@echo "                (add DROP_DB=1 to drop & recreate DB for a clean reload)"
	@echo "  psql        - Connect to DB_NAME"
	@echo "  shell       - Bash shell inside container"
	@echo "  down        - Stop container"
	@echo "  is-rootless - Check docker rootless mode"
	@echo "  help        - This message"
	@echo ""
	@echo "Variables (override on the command line):"
	@echo "  DATASET   = $(DATASET)   (dir under repo data/, = /datasets/<DATASET>)"
	@echo "  DB_NAME   = $(DB_NAME)   (auto-created if missing)"
	@echo "  DATA_DIR  = $(DATA_DIR)"
	@echo "  TPCH_DIR / REVIEWS_FILE / IMAGES_FILE / REVIEWS_Q_FILE / IMAGES_Q_FILE"
	@echo ""
	@echo "Examples:"
	@echo "  make load-data                                  # defaults"
	@echo "  make load-data DATASET=vech-sf1 DB_NAME=vech_sf1"
	@echo "  make load-data DROP_DB=1                         # clean reload"
	@echo "  make psql DB_NAME=vech_sf1"
