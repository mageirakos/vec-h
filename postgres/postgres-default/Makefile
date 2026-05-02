ROOTLESS := $(shell docker info --format '{{json .SecurityOptions}}' | grep -q rootless && echo 1 || echo 0)
IS_ROOTLESS := $(if $(filter 1,$(ROOTLESS)),true,false)

SF ?= 1

# TODO: adjust these paths to match default filesystem layout
TPCH_DIR ?= /datasets/tpch-datasets/csv/csv-$(SF)
REVIEWS_FILE ?= /datasets/industrial_and_scientific_sf$(SF)_reviews.parquet
IMAGES_FILE ?= /datasets/industrial_and_scientific_sf$(SF)_images.parquet
REVIEWS_Q_FILE ?= /datasets/industrial_and_scientific_sf$(SF)_reviews_queries.parquet
IMAGES_Q_FILE ?= /datasets/industrial_and_scientific_sf$(SF)_images_queries.parquet

.PHONY: setup down shell psql load-data help is-rootless

setup:
	mkdir -p ~/postgres-scratch/pg_data
	docker compose up -d --build
	@echo ""
	@echo "Postgres starting. Check: docker compose logs -f"
	@echo "Next: make load-data SF=$(SF)"

down:
	docker compose down

shell:
	docker compose exec pgvector bash

psql:
	docker compose exec pgvector psql -U postgres -d vech

load-data:
	docker compose exec pgvector python3 /maxvec/load_to_postgres_database.py \
		--sf $(SF) \
		--db_name vech \
		--db_host localhost \
		--db_user postgres \
		--db_password 1234 \
		--tpch_dir $(TPCH_DIR) \
		--reviews_file $(REVIEWS_FILE) \
		--images_file $(IMAGES_FILE) \
		--reviews_queries_file $(REVIEWS_Q_FILE) \
		--images_queries_file $(IMAGES_Q_FILE)
	@echo ""
	@echo "Running ANALYZE..."
	docker compose exec pgvector psql -U postgres -d vech -c "ANALYZE;"
	@echo "Done. Verify: make psql"

is-rootless:
	@echo "Docker rootless: $(IS_ROOTLESS)"

help:
	@echo "Targets:"
	@echo "  setup          - Build image + start postgres (creates ~/postgres-scratch/)"
	@echo "  load-data SF=1 - Load TPC-H + VECH data (run once per SF)"
	@echo "  psql           - Connect to vech database"
	@echo "  shell          - Bash shell inside container"
	@echo "  down           - Stop container"
	@echo "  is-rootless    - Check docker rootless mode"
	@echo "  help           - This message"
