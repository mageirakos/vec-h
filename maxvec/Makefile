UID := $(shell id -u)
GID := $(shell id -g)

ROOTLESS := $(shell docker info --format '{{json .SecurityOptions}}' | grep -q rootless && echo 1 || echo 0)

ifeq ($(ROOTLESS),1)
  USER_FLAG :=
else
  USER_FLAG := $(UID):$(GID)
endif

export UID GID

IS_ROOTLESS := $(if $(filter 1,$(ROOTLESS)),true,false)

IMAGE := maxvs-dev-gpu
SERVICE := maximus
WORKDIR := /home/Maximus

# Test image/container (separate from the main one)
# You can make a change in the Dockerfile and make test-setup --> it will build new container  (just make sure you revert any changes in the dockerfile you don't want. )
TEST_IMAGE := test-maxvs-dev-gpu
TEST_CONTAINER := test-maxvs

# Build configuration: Release (default) or Debug
BUILD_TYPE ?= Release
BUILD_DIR := build/$(BUILD_TYPE)

# Default optimization : avx512 or sve - can be overridden: make setup OPT_LEVEL=avx2 PARALLEL_JOBS=16
OPT_LEVEL ?= auto
PARALLEL_JOBS ?= 8
BUILD_TYPE ?= Release
# ============================================================================
# CAGRA Patches (both default false — base container has no patches)
#
# Workflow option 1: Build with patches baked in (one step):
#   make setup APPLY_RAFT_ATS_PATCH=true APPLY_FAISS_CAGRA_PATCH=true APPLY_FAISS_IVF_PATCH=true
#
# Workflow option 2: Build base, then patch at runtime (three steps):
#   make setup                          # clean base container, no patches
#   make apply-raft-cuvs-ats-patch     # patches headers + rebuilds Faiss + Maximus
#   make apply-faiss-cagra-patch       # patches Faiss source + rebuilds Faiss + Maximus
#
# Both workflows produce the same binary. Option 1 is faster, option 2 is
# more flexible (e.g., test without patches first, then add them).
#
# Raft patch: runtime data=0/data=1 control for CAGRA on ATS systems.
# Faiss patch: cagra_cache_graph=1 for fast to_gpu() with cached graph.
# See third_party/patches/README.md for details.
# ============================================================================
APPLY_RAFT_ATS_PATCH ?= false
APPLY_FAISS_CAGRA_PATCH ?= false
APPLY_FAISS_IVF_PATCH ?= false
# Test container defaults to RelWithDebInfo for debug symbols; override with BUILD_TYPE=Release if needed
TEST_BUILD_TYPE ?= RelWithDebInfo


.PHONY: compose-up compose-down shell build-maximus clean-build rebuild-faiss clone-faiss logs build-image setup reset test reset-test is_rootless restart test-build-image test-shell test-container-down apply-raft-cuvs-ats-patch apply-faiss-cagra-patch apply-faiss-ivf-patch check-patches full-run run-vsds run-varbatch recall-run kill-run

setup: build-image compose-down compose-up 
	@echo "Waiting for $(SERVICE)..."
	@# Try 30 times, then fail
	@for i in `seq 1 30`; do \
	    if docker compose exec -T $(SERVICE) true >/dev/null 2>&1; then \
	        echo "Container is ready."; \
	        $(MAKE) build-maximus; \
	        echo "Full setup complete."; \
			echo "use 'docker image prune -f' to clean up old unused <none>:<none> images."; \
	        exit 0; \
	    fi; \
	    sleep 1; \
	    echo -n "."; \
	done; \
	echo "Container failed to start."; exit 1

build-image:
	@echo "Building image with OPT_LEVEL=$(OPT_LEVEL) PARALLEL_JOBS=$(PARALLEL_JOBS) BUILD_TYPE=$(BUILD_TYPE) APPLY_RAFT_ATS_PATCH=$(APPLY_RAFT_ATS_PATCH) APPLY_FAISS_CAGRA_PATCH=$(APPLY_FAISS_CAGRA_PATCH) APPLY_FAISS_IVF_PATCH=$(APPLY_FAISS_IVF_PATCH)..."
	BUILD_TYPE=$(BUILD_TYPE) docker compose build --build-arg OPT_LEVEL=$(OPT_LEVEL) --build-arg PARALLEL_JOBS=$(PARALLEL_JOBS) --build-arg APPLY_RAFT_ATS_PATCH=$(APPLY_RAFT_ATS_PATCH) --build-arg APPLY_FAISS_CAGRA_PATCH=$(APPLY_FAISS_CAGRA_PATCH) --build-arg APPLY_FAISS_IVF_PATCH=$(APPLY_FAISS_IVF_PATCH)

restart:
	@echo "Restarting compose..."
	$(MAKE) compose-down && $(MAKE) compose-up

compose-up:
	@echo "Starting compose (USER_FLAG=$(USER_FLAG) )..."
	docker compose up -d

compose-down:
	docker compose down

is_rootless:
	@echo "Docker Rootless mode: $(IS_ROOTLESS)"
	@if [ "$(IS_ROOTLESS)" = "true" ]; then \
		echo "----> You will be root inside the container. It is okay. <----"; \
	else \
		echo "----> You should NOT be root inside the container! Always use make shell <----"; \
	fi

ensure-running:
	@if ! docker compose ps $(SERVICE) --format json | grep -q '"State":"running"'; then \
		$(MAKE) compose-up; \
	fi

shell: ensure-running
	@echo "Starting shell in $(SERVICE) Rootless mode = $(IS_ROOTLESS)..."
	docker compose exec $(if $(USER_FLAG),--user $(USER_FLAG)) $(SERVICE) bash

run: ensure-running
	@docker compose exec $(if $(USER_FLAG),--user $(USER_FLAG)) -w $(WORKDIR) $(SERVICE) bash -lc "$(CMD)"

# --- Full experiment run targets ---
SYSTEM ?= dgx-spark-02
SF ?= 1
PERSIST_CSV ?= true
STAGES ?= all
ANALYZE_ONLY ?=
GPU_ID ?= 0

# Translate Makefile vars into full_run.sh flags
FULL_RUN_FLAGS = --sf=$(SF)
ifneq ($(STAGES),all)
  FULL_RUN_FLAGS += --stages=$(STAGES)
endif
ifneq ($(ANALYZE_ONLY),)
  FULL_RUN_FLAGS += --analyze-only
endif
ifneq ($(PERSIST_CSV),true)
  FULL_RUN_FLAGS += --no-csv
endif

# Date tag: 22Mar26 format (force C locale for consistent month names)
DATE_TAG = $(or $(RUN_DATE),$(shell LC_TIME=C date +%d%b%y))

# Auto-detect next run ID for today's date + system combo
RUN_DIR_BASE = results/runs/$(DATE_TAG)-$(SYSTEM)
AUTO_RUN_ID = $(shell \
    if [ -d "$(RUN_DIR_BASE)" ]; then \
        max=$$(ls -d "$(RUN_DIR_BASE)"/run* 2>/dev/null | sed 's|.*/run||' | sort -n | tail -1); \
        echo $$(( $${max:-0} + 1 )); \
    else \
        echo 1; \
    fi)
RUN_ID ?= $(AUTO_RUN_ID)
FULL_RUN_DIR = results/runs/$(DATE_TAG)-$(SYSTEM)/run$(RUN_ID)

# Full experiment run: benchmarks + parse, detached (survives SSH disconnect).
# Kills any prior run first (idempotent).
#
#   make full-run SYSTEM=sgs-gpu06                           # auto-increments run ID
#   make full-run SYSTEM=sgs-gpu06 RUN_ID=2                  # force specific run ID
#   make full-run SYSTEM=sgs-gpu06 SF=0.01                   # custom scale factor
#   make full-run SYSTEM=sgs-gpu06 STAGES=vsds               # only VSDS
#   make full-run SYSTEM=sgs-gpu06 STAGES=vsds,varbatch      # two stages
#   make full-run SYSTEM=sgs-gpu06 ANALYZE_ONLY=1 STAGES=vsds,varbatch RUN_ID=3
#                                                            # re-parse+plot existing logs (no benchmarks)
#   make full-run SYSTEM=sgs-gpu06 PERSIST_CSV=false         # skip CSV + recall
full-run:
	@$(MAKE) kill-run
	@$(MAKE) build-maximus
	@echo "============================================"
	@echo "  FULL RUN"
	@echo "  System:       $(SYSTEM)"
	@echo "  GPU ID:       $(GPU_ID)"
	@echo "  SF:           $(SF)"
	@echo "  Date:         $(DATE_TAG)"
	@echo "  Run ID:       $(RUN_ID)"
	@echo "  Stages:       $(STAGES)"
	@echo "  Analyze only: $(if $(ANALYZE_ONLY),yes,no)"
	@echo "  Persist CSV:  $(PERSIST_CSV)"
	@echo "  Output:       $(FULL_RUN_DIR)/"
	@echo "  Flags:        $(FULL_RUN_FLAGS)"
	@echo "============================================"
	@mkdir -p $(FULL_RUN_DIR)
	docker compose exec -d \
		$(if $(USER_FLAG),--user $(USER_FLAG)) \
		-e CUDA_VISIBLE_DEVICES=$(GPU_ID) \
		-e MAXIMUS_OPERATORS_FUSION=0 \
		-e MAXIMUS_DATASET_API=0 \
		-w $(WORKDIR) \
		$(SERVICE) bash -lc \
		"./results/full_run.sh ./$(FULL_RUN_DIR) $(SYSTEM) $(FULL_RUN_FLAGS)"
	@echo ""
	@echo "Experiment launched in background."
	@echo "Monitor: make run CMD='tail -f $(FULL_RUN_DIR)/full_run.log'"
	@echo "Check:   make run CMD='ps aux | grep full_run'"

# Run just VSDS benchmarks (configurable, detached)
#   make run-vsds SYSTEM=sgs-gpu06 SF=0.01 VSDS_Q=q1_c0
VSDS_Q ?= all
VSDS_VS ?= all
VSDS_K ?= 100
run-vsds: build-maximus
	@echo "Running VSDS: SF=$(SF) Q=$(VSDS_Q) VS=$(VSDS_VS) K=$(VSDS_K) SYSTEM=$(SYSTEM)"
	@echo "Output: $(FULL_RUN_DIR)/"
	@mkdir -p $(FULL_RUN_DIR)
	docker compose exec -d \
		$(if $(USER_FLAG),--user $(USER_FLAG)) \
		-e CUDA_VISIBLE_DEVICES=$(GPU_ID) \
		-e MAXIMUS_OPERATORS_FUSION=0 \
		-e MAXIMUS_DATASET_API=0 \
		-w $(WORKDIR) \
		$(SERVICE) bash -lc \
		"./results/run_vsds.sh $(VSDS_Q) $(SF) $(VSDS_VS) $(SYSTEM) $(VSDS_K) ./$(FULL_RUN_DIR) \
		 2>&1 | tee -a ./$(FULL_RUN_DIR)/run_vsds.log"
	@echo "VSDS launched. Monitor: make run CMD='tail -f $(FULL_RUN_DIR)/run_vsds.log'"

# Run just other experiments / varbatch (configurable, detached)
#   make run-varbatch SYSTEM=sgs-gpu06 SF=1 OTHER_VS=vary-batch-in-maxbench
OTHER_VS ?= vary-batch-in-maxbench
OTHER_CASE ?= all
run-varbatch: build-maximus
	@echo "Running other experiments: SF=$(SF) VS=$(OTHER_VS) SYSTEM=$(SYSTEM)"
	@echo "Output: $(FULL_RUN_DIR)/"
	@mkdir -p $(FULL_RUN_DIR)
	docker compose exec -d \
		$(if $(USER_FLAG),--user $(USER_FLAG)) \
		-e CUDA_VISIBLE_DEVICES=$(GPU_ID) \
		-e MAXIMUS_OPERATORS_FUSION=0 \
		-e MAXIMUS_DATASET_API=0 \
		-w $(WORKDIR) \
		$(SERVICE) bash -lc \
		"./results/run_other_experiments.sh $(SF) $(OTHER_VS) $(SYSTEM) $(OTHER_CASE) ./$(FULL_RUN_DIR) \
		 2>&1 | tee -a ./$(FULL_RUN_DIR)/run_other.log"
	@echo "Launched. Monitor: make run CMD='tail -f $(FULL_RUN_DIR)/run_other.log'"

# Kill all running experiment processes inside the container
#   make kill-run
kill-run: ensure-running
	@echo "Killing experiment processes inside container..."
	@docker compose exec $(SERVICE) bash -c \
		'pids=$$(pgrep -f "maxbench|run_vsds|run_other|full_run" 2>/dev/null | grep -vw $$$$); \
		 if [ -n "$$pids" ]; then \
		   echo "Killing PIDs: $$pids"; \
		   kill $$pids 2>/dev/null; \
		   sleep 1; \
		   kill -9 $$pids 2>/dev/null; \
		   echo "Done."; \
		 else \
		   echo "No experiment processes found."; \
		 fi'

# Re-analyze a run (parse + plot, no benchmarks):
#   make full-run SYSTEM=cscs-gh200 RUN_DATE=27Mar26 RUN_ID=5_gpu0 ANALYZE_ONLY=1
#   make full-run SYSTEM=cscs-gh200 RUN_DATE=27Mar26 RUN_ID=5_gpu0 ANALYZE_ONLY=1 STAGES=vsds,varbatch

# Calculate recall for a run (requires CSV output from benchmarks).
#   make recall-run SYSTEM=cscs-gh200 RUN_ID=5_gpu0 RUN_DATE=27Mar26
#   make recall-run SYSTEM=cscs-gh200 RUN_ID=5_gpu0 RUN_DATE=27Mar26 SF=0.01
recall-run:
	@echo "Calculating recall: $(FULL_RUN_DIR)"
	cd results && \
	echo "  recall (vsds)..." && \
	python recall_vsds.py --results_dir ../$(FULL_RUN_DIR)/vsds \
		--sf $(SF) --system $(SYSTEM) \
		--recall_dir ../$(FULL_RUN_DIR)/recall/vsds \
		--out $(SYSTEM)_vsds_recall_sf_$(SF).csv \
	|| echo "  [WARN] recall vsds failed"; \
	echo "  recall (vary-batch)..." && \
	python recall_vsds.py --results_dir ../$(FULL_RUN_DIR)/other-vary-batch-in-maxbench \
		--sf $(SF) --system $(SYSTEM) \
		--recall_dir ../$(FULL_RUN_DIR)/recall/vary-batch \
		--out $(SYSTEM)_vary-batch_recall_sf_$(SF).csv \
		--batch_filter 10000 \
	|| echo "  [WARN] recall vary-batch failed"

build-maximus: ensure-running
	docker compose exec $(if $(USER_FLAG),--user $(USER_FLAG)) -w $(WORKDIR) $(SERVICE) bash -lc "\
	    mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
	    cmake ../.. -G Ninja \
	        -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	        -DMAXIMUS_WITH_TESTS=ON \
	        -DMAXIMUS_WITH_GPU=ON \
	        -DMAXIMUS_WITH_FAISS=GPUCUVS \
	        -DMAXIMUS_WITH_VS=ON \
	        -DMAXIMUS_WITH_PROFILING=ON \
	        -DCMAKE_PREFIX_PATH=\"/usr/local\" && \
	    ninja"

clean-build: ensure-running
	docker compose exec $(if $(USER_FLAG),--user $(USER_FLAG)) -w $(WORKDIR) $(SERVICE) bash -lc "rm -rf $(BUILD_DIR)"

# Rebuild Faiss inside the container after source edits.
# Source lives at /home/faiss (preserved across container restarts).
# devuser owns both /home/faiss and $CONDA_PREFIX (set in Dockerfile), so no sudo needed.
# If /home/faiss doesn't exist (older image), clone it first with: make clone-faiss
rebuild-faiss: ensure-running
	docker compose exec $(if $(USER_FLAG),--user $(USER_FLAG)) $(SERVICE) bash -lc "\
	    test -d /home/faiss/build || { echo 'ERROR: /home/faiss/build not found. Run: make clone-faiss'; exit 1; } && \
	    ninja -C /home/faiss/build -j $(PARALLEL_JOBS) faiss swigfaiss && \
	    ninja -C /home/faiss/build install"

# Clone + configure Faiss source inside the container (one-time, for images that didn't preserve it).
# On new images (Dockerfile step 12), devuser owns $CONDA_PREFIX, so no sudo needed.
# On older images, falls back to sudo for the clone + chown.
clone-faiss: ensure-running
	docker compose exec $(if $(USER_FLAG),--user $(USER_FLAG)) $(SERVICE) bash -lc "\
	    if [ -d /home/faiss ]; then echo 'Faiss source already at /home/faiss'; exit 0; fi && \
	    sudo git clone --depth 1 https://github.com/facebookresearch/faiss.git --branch v1.13.0 /home/faiss && \
	    sudo chown -R \$$(id -u):\$$(id -g) /home/faiss && \
	    ARCH=\$$(uname -m) && \
	    if [ \"\$$ARCH\" = 'x86_64' ]; then OPT=avx512; else OPT=sve; fi && \
	    echo \"Configuring Faiss for \$$ARCH with OPT=\$$OPT\" && \
	    cmake -B /home/faiss/build /home/faiss -G Ninja \
	        -DFAISS_ENABLE_GPU=ON \
	        -DFAISS_ENABLE_CUVS=ON \
	        -DFAISS_ENABLE_PYTHON=ON \
	        -DFAISS_OPT_LEVEL=\$$OPT \
	        -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	        -DBLA_VENDOR=OpenBLAS \
	        -DPython_EXECUTABLE=\$$CONDA_PREFIX/bin/python \
	        -DCMAKE_INSTALL_PREFIX=\$$CONDA_PREFIX"

# Check whether both CAGRA patches are applied in the running container.
# Raft patch: header-only (inline atomic in raft/neighbors/dataset.hpp) — check header content.
#   Gets compiled into libfaiss.a when Faiss is built (CuvsCagra.cu includes these headers).
# Faiss patch: adds copyFrom_graph method to Faiss source — check both source and libfaiss.a symbol.
check-patches: ensure-running
	@echo "=== Raft ATS runtime patch ==="
	@docker compose exec $(if $(USER_FLAG),--user $(USER_FLAG)) $(SERVICE) bash -lc \
	    "cd \$$CONDA_PREFIX/include && patch -p0 --forward --dry-run < /home/Maximus/third_party/patches/raft-ats-runtime.patch > /dev/null 2>&1 \
	    && echo '  Headers: NOT applied' || echo '  Headers: Applied'"
	@docker compose exec $(if $(USER_FLAG),--user $(USER_FLAG)) $(SERVICE) bash -lc \
	    "nm \$$CONDA_PREFIX/lib/libfaiss.a 2>/dev/null | grep -q cagra_ats_force_copy_enabled \
	    && echo '  Compiled into libfaiss.a: Yes' || echo '  Compiled into libfaiss.a: No (rebuild Faiss after patching)'"
	@echo "=== Faiss CAGRA graph cache patch ==="
	@docker compose exec $(if $(USER_FLAG),--user $(USER_FLAG)) $(SERVICE) bash -lc \
	    "cd /home/faiss && patch -p0 --forward --dry-run < /home/Maximus/third_party/patches/faiss-cagra-graph.patch > /dev/null 2>&1 \
	    && echo '  Source: NOT applied' || echo '  Source: Applied'"
	@docker compose exec $(if $(USER_FLAG),--user $(USER_FLAG)) $(SERVICE) bash -lc \
	    "nm \$$CONDA_PREFIX/lib/libfaiss.a 2>/dev/null | grep -q copyFrom_graph \
	    && echo '  Compiled into libfaiss.a: Yes' || echo '  Compiled into libfaiss.a: No'"
	@echo "=== Faiss IVF host view patch ==="
	@docker compose exec $(if $(USER_FLAG),--user $(USER_FLAG)) $(SERVICE) bash -lc \
	    "cd /home/faiss && patch -p0 --forward --dry-run < /home/Maximus/third_party/patches/faiss-ivf-host-view.patch > /dev/null 2>&1 \
	    && echo '  Source: NOT applied' || echo '  Source: Applied'"
	@docker compose exec $(if $(USER_FLAG),--user $(USER_FLAG)) $(SERVICE) bash -lc \
	    "nm \$$CONDA_PREFIX/lib/libfaiss.a 2>/dev/null | grep -q referenceInvertedListsFrom \
	    && echo '  Compiled into libfaiss.a: Yes' || echo '  Compiled into libfaiss.a: No'"

# Apply raft/cuVS ATS runtime patch to headers inside the container.
# Adds std::atomic<bool> flag for runtime data=0/data=1 control.
apply-raft-cuvs-ats-patch: ensure-running
	docker compose exec $(if $(USER_FLAG),--user $(USER_FLAG)) $(SERVICE) bash -lc \
	    "cd \$$CONDA_PREFIX/include && patch -p0 --forward < /home/Maximus/third_party/patches/raft-ats-runtime.patch"
	$(MAKE) rebuild-faiss
	$(MAKE) build-maximus

# Apply Faiss CAGRA graph cache patch (adds copyFrom_graph + uint32 constructor).
# Enables cagra_cache_graph=1 for fast to_gpu() with cached graph.
apply-faiss-cagra-patch: ensure-running
	docker compose exec $(if $(USER_FLAG),--user $(USER_FLAG)) $(SERVICE) bash -lc \
	    "cd /home/faiss && patch -p0 --forward < /home/Maximus/third_party/patches/faiss-cagra-graph.patch"
	$(MAKE) rebuild-faiss
	$(MAKE) build-maximus

# Apply Faiss IVF host view patch (adds referenceFrom + referenceInvertedListsFrom).
# Enables index_data_on_gpu=0 for IVF indexes on ATS systems (zero-copy host view).
apply-faiss-ivf-patch: ensure-running
	docker compose exec $(if $(USER_FLAG),--user $(USER_FLAG)) $(SERVICE) bash -lc \
	    "cd /home/faiss && patch -p0 --forward < /home/Maximus/third_party/patches/faiss-ivf-host-view.patch"
	$(MAKE) rebuild-faiss
	$(MAKE) build-maximus

reset: clean-build build-maximus
	@echo "Reset complete. Project is clean and rebuilt."

test: ensure-running
	docker compose exec $(if $(USER_FLAG),--user $(USER_FLAG)) -w $(WORKDIR)/$(BUILD_DIR) $(SERVICE) bash -lc "ninja test"

reset-test: reset test
	@echo "Full reset, build, and test completed."

build-test: build-maximus test
	@echo "Build and test completed."

logs:
	docker compose logs -f $(SERVICE)

# --- Test image targets (separate container, does not touch the main one) ---
# Build the test image with a different tag
test-build-image:
	@echo "Building test image $(TEST_IMAGE) with OPT_LEVEL=$(OPT_LEVEL) PARALLEL_JOBS=$(PARALLEL_JOBS) BUILD_TYPE=$(TEST_BUILD_TYPE)..."
	BUILD_TYPE=$(TEST_BUILD_TYPE) docker compose build --build-arg OPT_LEVEL=$(OPT_LEVEL) --build-arg PARALLEL_JOBS=$(PARALLEL_JOBS) test-maximus

# Start a test container (GPU, same config as main service via compose)
test-container-up: test-build-image
	@echo "Starting test container $(TEST_CONTAINER)..."
	docker compose --profile test up -d test-maximus

test-container-down:
	@echo "Stopping and removing test container $(TEST_CONTAINER)..."
	docker compose --profile test down test-maximus

# Shell into the test container
test-shell:
	docker compose --profile test exec $(if $(USER_FLAG),--user $(USER_FLAG)) test-maximus bash

# Full test setup: build image + start container
test-setup: test-container-up
	@echo "Test container $(TEST_CONTAINER) is running. Use 'make test-shell' to enter it."

# Helper target to show available options
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make setup                   - Full setup with default options (OPT_LEVEL=auto)"
	@echo "  make setup OPT_LEVEL=avx2    - Setup with specific CPU optimization"
	@echo "  make build-image             - Build Docker image only"
	@echo "  make shell                   - Enter container shell"
	@echo "  make build-maximus           - Build Maximus (defaults to Release)"
	@echo "  make build-maximus BUILD_TYPE=Debug - Build in Debug mode"
	@echo "  make test                    - Run tests (defaults to Release)"
	@echo "  make test BUILD_TYPE=Debug   - Run tests for Debug build"
	@echo "  make reset                   - Clean and rebuild current build type"
	@echo "  make logs                    - Show container logs"
	@echo ""
	@echo "Experiment run targets (detached, survives SSH disconnect):"
	@echo "  full-run (kills any prior run first, then launches detached):"
	@echo "    Flags (set as VAR=value on make command line):"
	@echo "      SYSTEM         System name (default: $(SYSTEM))"
	@echo "      SF             Scale factor (default: $(SF))"
	@echo "      GPU_ID         CUDA_VISIBLE_DEVICES (default: $(GPU_ID))"
	@echo "      RUN_ID         Override run ID (default: auto-increment)"
	@echo "      RUN_DATE       Override date tag (default: today)"
	@echo "      STAGES         Comma-separated: vsds,varbatch,fullsweep (default: all)"
	@echo "      ANALYZE_ONLY   Set to any value to skip build+benchmarks (parse+plot+recall only)"
	@echo "      PERSIST_CSV    'true' (default) writes CSVs + runs recall; 'false' skips both"
	@echo ""
	@echo "    Examples:"
	@echo "      make full-run SYSTEM=sgs-gpu06                             - default: all stages"
	@echo "      make full-run SYSTEM=sgs-gpu06 STAGES=vsds                 - VSDS only"
	@echo "      make full-run SYSTEM=sgs-gpu06 STAGES=varbatch PERSIST_CSV=false"
	@echo "                                                                 - varbatch only, no CSV/recall"
	@echo "      make full-run SYSTEM=sgs-gpu06 STAGES=fullsweep            - bs1_fullsweep (CASE 6) only"
	@echo "      make full-run SYSTEM=sgs-gpu06 STAGES=vsds,varbatch        - two stages"
	@echo "      make full-run SYSTEM=sgs-gpu06 SF=0.01                     - custom scale factor"
	@echo "      make full-run SYSTEM=sgs-gpu06 PERSIST_CSV=false           - skip CSV + recall"
	@echo "      make full-run SYSTEM=sgs-gpu06 RUN_ID=3 ANALYZE_ONLY=1 STAGES=vsds,varbatch"
	@echo "                                                                    (crash recovery: re-parse+plot existing logs)"
	@echo ""
	@echo "  Other experiment targets:"
	@echo "    make run-vsds SYSTEM=sgs-gpu06           - VSDS only (configurable: VSDS_Q, VSDS_VS, VSDS_K)"
	@echo "    make run-varbatch SYSTEM=sgs-gpu06       - Varbatch only (configurable: OTHER_VS, OTHER_CASE)"
	@echo "    make recall-run SYSTEM=cscs-gh200 RUN_DATE=27Mar26 RUN_ID=5_gpu0 - Calculate recall only"
	@echo "    make kill-run                             - Kill all running experiment processes in container"
	@echo ""
	@echo "  Monitor: make run CMD='tail -f results/runs/<date>-<system>/run<id>/full_run.log'"
	@echo "  Output:  results/runs/<date>-<system>/run<id>/"
	@echo ""
	@echo "Build arguments:"
	@echo "  OPT_LEVEL     - CPU optimization level (default: auto)"
	@echo "                  Options: avx2, avx512, sve, generic"
	@echo "  PARALLEL_JOBS - Number of parallel build jobs (default: 8)"
	@echo ""
	@echo "Examples:"
	@echo "  make setup OPT_LEVEL=avx512 PARALLEL_JOBS=16"
	@echo "  make setup APPLY_RAFT_ATS_PATCH=true APPLY_FAISS_CAGRA_PATCH=true APPLY_FAISS_IVF_PATCH=true"
	@echo "  make build-image OPT_LEVEL=generic"