IMAGE := pgvector-cscs
CONTEXT := ../postgres-scripts

.PHONY: build-image save-image deploy help

build-image:
	docker build -t $(IMAGE) $(CONTEXT)

save-image: build-image
	docker save $(IMAGE) | gzip > $(IMAGE).tar.gz
	@echo "Saved: $(IMAGE).tar.gz ($(shell du -h $(IMAGE).tar.gz | cut -f1))"

deploy:
	@echo "=== Deploy pgvector image to CSCS ==="
	@echo ""
	@echo "1. Transfer image:"
	@echo "   rsync -avP $(IMAGE).tar.gz daint:\$$SCRATCH/"
	@echo ""
	@echo "2. On daint (interactive session):"
	@echo "   srun -A sm94 --pty bash"
	@echo "   mkdir -p \$$SCRATCH/ce-images"
	@echo "   gunzip -c \$$SCRATCH/$(IMAGE).tar.gz | podman load"
	@echo "   enroot import -x mount -o \$$SCRATCH/ce-images/pgvector.sqsh podman://$(IMAGE):latest"
	@echo ""
	@echo "3. Create EDF: ~/.edf/pgvector.toml (coexists with any other ~/.edf/*.toml)"
	@echo '   image = "$${SCRATCH}/ce-images/pgvector.sqsh"'
	@echo '   mounts = ["/capstor", "/iopsstor", "/users"]'
	@echo '   workdir = "/users/$${USER}"'
	@echo "   See postgres-scripts/NEW_CONTAINER_TESTING.md (EDF / mounts / persistence) for details."
	@echo ""
	@echo "4. Create scratch dirs:"
	@echo "   mkdir -p \$$SCRATCH/pg_data \$$SCRATCH/datasets"
	@echo ""
	@echo "5. Load data:"
	@echo "   sbatch setup_pgvector_cscs.sh"

help:
	@echo "Targets:"
	@echo "  build-image  - Build Docker image from ../postgres-scripts/"
	@echo "  save-image   - Build + save as .tar.gz"
	@echo "  deploy       - Print CSCS deployment instructions"
	@echo "  help         - This message"
