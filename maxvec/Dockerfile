# Universal Dockerfile for x86_64 (AMD/Intel) and aarch64 (Grace/DGX Spark)
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]
WORKDIR /home

# Optimization Level: "auto" detects the arch. 
# Override with: --build-arg OPT_LEVEL="avx2" if needed.
ARG OPT_LEVEL="auto"
ARG PARALLEL_JOBS=8
# Use Release or RelWithDebInfo
ARG BUILD_TYPE="Release"

RUN echo "Building for architecture: $(uname -m) with OPT_LEVEL=${OPT_LEVEL} and BUILD_TYPE=${BUILD_TYPE}"

# 1. Install Basic Utils & Debugging Tools
RUN apt-get update && apt-get install -y \
    wget git build-essential ninja-build \
    ca-certificates pkg-config \
    sudo clang-format \
    libssl-dev zlib1g-dev \
    vim \
    numactl \
    cuda-nsight-systems-12-8 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Miniforge (Dynamic Arch)
# Automatically downloads the correct installer for x86 or ARM.
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    wget "$MINIFORGE_URL" -O miniforge.sh && \
    bash miniforge.sh -b -p /opt/conda && \
    rm miniforge.sh
    
ENV PATH="/opt/conda/bin:$PATH"


# 3. Create Environment (Dynamic Compilers)
# If x86, install gcc_linux-64. If ARM, install gcc_linux-aarch64.
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        COMPILERS="gcc_linux-64 gxx_linux-64 sysroot_linux-64"; \
    else \
        COMPILERS="gcc_linux-aarch64 gxx_linux-aarch64 sysroot_linux-aarch64"; \
    fi && \
    echo "Installing compilers: $COMPILERS" && \
    mamba create -n maximus_env -y \
        -c rapidsai -c conda-forge -c nvidia \
        python=3.10 \
        # cuda has to be >12.8 for blackwell
        cuda-version=12.8 \ 
        "libcuvs=25.08" \
        "libraft=25.08" \
        "cudf=25.08" \
        cuda-nvtx-dev \
        $COMPILERS \
        gdb \
        cuda-gdb \
        cmake ninja pkg-config \
        swig \
        doxygen graphviz \
        sqlite \ 
        openblas=*=openmp* glog gflags \
        gtest gmock \
        boost-cpp \
        rapidjson \
        brotli \
        snappy \
        zlib \
        lz4-c \
        zstd \
        libprotobuf \
        pandas pyarrow duckdb \
        && mamba clean -afy

# 4. Activate Environment
ENV CONDA_DEFAULT_ENV=maximus_env
ENV CONDA_PREFIX="/opt/conda/envs/maximus_env"
ENV PATH="${CONDA_PREFIX}/bin:$PATH"
ENV LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH"

# Install pure-python libs (explicitly use maximus_env pip)
RUN  ${CONDA_PREFIX}/bin/pip install --no-cache-dir psycopg pgvector tabulate openpyxl matplotlib seaborn plotly

# 5. Build and install rapids-cmake from source
RUN git clone --depth 1 --branch branch-25.08 https://github.com/rapidsai/rapids-cmake.git /tmp/rc && \
    cmake -S /tmp/rc -B /tmp/rc/build -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX && \
    cmake --install /tmp/rc/build && \
    rm -rf /tmp/rc

# 8. Build Apache Arrow 21.0.0 (Custom Configuration)
# Adapted from your bash script.
# We install to $CONDA_PREFIX so CMake/pkg-config find it automatically.
RUN git clone --recursive https://github.com/apache/arrow --branch apache-arrow-21.0.0 /tmp/arrow && \
    cd /tmp/arrow/cpp && \
    mkdir build && cd build && \
    # Explicitly using the compilers from the Conda env
    # NOTE: if you want to have debugging flags use build type: RelWithDebInfo
    CC=gcc CXX=g++ cmake .. -G Ninja \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \     
        -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
        -Dxsimd_SOURCE=BUNDLED \
        -DARROW_WITH_UTF8PROC=OFF \
        -DARROW_WITH_RE2=ON \
        -DARROW_FILESYSTEM=ON \
        -DARROW_TESTING=OFF \
        -DARROW_WITH_LZ4=OFF \
        -DARROW_WITH_ZSTD=OFF \
        -DARROW_BUILD_STATIC=ON \
        -DARROW_BUILD_SHARED=ON \
        -DARROW_BUILD_TESTS=OFF \
        -DARROW_BUILD_BENCHMARKS=OFF \
        -DARROW_IPC=ON \
        -DARROW_FLIGHT=OFF \
        -DARROW_COMPUTE=ON \
        -DARROW_CUDA=OFF \
        -DARROW_JEMALLOC=OFF \
        -DARROW_USE_GLOG=OFF \
        -DARROW_DATASET=ON \
        -DARROW_BUILD_UTILITIES=OFF \
        -DARROW_HDFS=OFF \
        -DCMAKE_VERBOSE_MAKEFILE=ON \
        -DARROW_TENSORFLOW=OFF \
        -DARROW_CSV=ON \
        -DARROW_JSON=ON \
        -DARROW_WITH_BROTLI=ON \
        -DARROW_WITH_SNAPPY=ON \
        -DARROW_WITH_ZLIB=ON \
        -DARROW_PARQUET=ON \
        -DARROW_SUBSTRAIT=OFF \
        -DARROW_ACERO=ON \
        -DARROW_WITH_BACKTRACE=ON \
        -DARROW_CXXFLAGS="-w" \
        -DARROW_S3=OFF \
        -DARROW_ORC=OFF \
        -DARROW_POSITION_INDEPENDENT_CODE=ON \
        -DARROW_DEPENDENCY_USE_SHARED=ON \
        -DARROW_BOOST_USE_SHARED=ON \
        -DARROW_BROTLI_USE_SHARED=ON \
        -DARROW_GFLAGS_USE_SHARED=ON \
        -DARROW_GRPC_USE_SHARED=ON \
        -DARROW_PROTOBUF_USE_SHARED=ON \
        -DARROW_ZSTD_USE_SHARED=ON \
        -DProtobuf_SOURCE=SYSTEM && \
    ninja install -j ${PARALLEL_JOBS} && \
    rm -rf /tmp/arrow

# 6. Install Taskflow
RUN git clone --depth 1 --branch v3.6.0 https://github.com/taskflow/taskflow.git /opt/taskflow && \
    cmake -S /opt/taskflow -B /opt/taskflow/build -G Ninja \
        -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
        -DTF_BUILD_TESTS=OFF \
        -DTF_BUILD_EXAMPLES=OFF && \
    cmake --install /opt/taskflow/build && \
    rm -rf /opt/taskflow

# 7. Build Caliper
# Installed directly into $CONDA_PREFIX so CMake finds it automatically.
RUN git clone https://github.com/LLNL/Caliper.git caliper && \
    cd caliper && \
    cmake -B build -G Ninja . \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
        -DWITH_CUDA=OFF \
        -DCUDAToolkit_ROOT=/usr/local/cuda \
        -DWITH_NVTX=ON \
        -DBUILD_TESTING=OFF \
        -DBUILD_DOCS=OFF && \
    ninja -C build -j ${PARALLEL_JOBS} install && \
    grep -q '^#define CALIPER_HAVE_NVTX' ${CONDA_PREFIX}/include/caliper/caliper-config.h && \
    cd .. && rm -rf caliper

# 8b. Optional: Apply raft/cuVS ATS runtime patch
# Adds std::atomic<bool> flag to make_strided_dataset() for runtime data=0/data=1 control.
# Must run BEFORE faiss build since faiss compiles CuvsCagra.cu which #includes these headers.
ARG APPLY_RAFT_ATS_PATCH=false
COPY third_party/patches/raft-ats-runtime.patch /tmp/raft-ats-runtime.patch
RUN if [ "$APPLY_RAFT_ATS_PATCH" = "true" ]; then \
        cd ${CONDA_PREFIX}/include && \
        patch -p0 --forward < /tmp/raft-ats-runtime.patch && \
        echo "Applied raft ATS runtime patch"; \
    fi

# 9. Build Faiss with Python bindings (Dynamic Optimization)
# NOTE: /home/faiss source + build dir are intentionally preserved so that
# Faiss can be patched and rebuilt at runtime via `make rebuild-faiss`.
# Do NOT add `rm -rf faiss` here.
ARG FAISS_VERSION=v1.13.0
RUN git clone --depth 1 https://github.com/facebookresearch/faiss.git --branch ${FAISS_VERSION} && \
    cd faiss && \
    ARCH=$(uname -m) && \
    if [ "${OPT_LEVEL}" = "auto" ]; then \
        if [ "$ARCH" = "x86_64" ]; then TARGET_OPT="avx512"; else TARGET_OPT="sve"; fi; \
    else \
        TARGET_OPT="${OPT_LEVEL}"; \
    fi && \
    echo "Building Faiss with OPT_LEVEL=$TARGET_OPT for ARCH=$ARCH" && \
    cmake -B build . -G Ninja \
        -DFAISS_ENABLE_GPU=ON \
        -DFAISS_ENABLE_CUVS=ON \
        -DFAISS_ENABLE_PYTHON=ON \
        -DFAISS_OPT_LEVEL=${TARGET_OPT} \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DBLA_VENDOR=OpenBLAS \
        -DPython_EXECUTABLE=${CONDA_PREFIX}/bin/python \
        -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} && \
    ninja -C build -j ${PARALLEL_JOBS} faiss swigfaiss && \
    ninja -C build install && \
    cd build/faiss/python && \
    ${CONDA_PREFIX}/bin/pip install --no-build-isolation . && \
    cd /home

# 9b. Optional: Apply Faiss CAGRA graph cache patch (adds copyFrom_graph())
# Must run AFTER faiss clone. If enabled, patches source and rebuilds.
ARG APPLY_FAISS_CAGRA_PATCH=false
COPY third_party/patches/faiss-cagra-graph.patch /tmp/faiss-cagra-graph.patch
RUN if [ "$APPLY_FAISS_CAGRA_PATCH" = "true" ]; then \
        cd /home/faiss && \
        patch -p0 --forward < /tmp/faiss-cagra-graph.patch && \
        ninja -C build -j ${PARALLEL_JOBS} faiss swigfaiss && \
        ninja -C build install && \
        echo "Applied Faiss CAGRA graph cache patch and rebuilt"; \
    fi

# 9c. Optional: Apply Faiss IVF host view patch (adds referenceFrom/referenceInvertedListsFrom)
# Enables index_data_on_gpu=0 for IVF indexes on ATS systems (zero-copy host view).
ARG APPLY_FAISS_IVF_PATCH=false
COPY third_party/patches/faiss-ivf-host-view.patch /tmp/faiss-ivf-host-view.patch
RUN if [ "$APPLY_FAISS_IVF_PATCH" = "true" ]; then \
        cd /home/faiss && \
        patch -p0 --forward < /tmp/faiss-ivf-host-view.patch && \
        ninja -C build -j ${PARALLEL_JOBS} faiss swigfaiss && \
        ninja -C build install && \
        echo "Applied Faiss IVF host view patch and rebuilt"; \
    else echo "Skipping Faiss IVF host view patch"; fi

# 10. Install Arrow GDB Pretty Printers (note: we might have to build arrow with RelWithDebInfo for this to work fully)
RUN mkdir -p ${CONDA_PREFIX}/share/arrow/gdb && \
    wget https://raw.githubusercontent.com/apache/arrow/apache-arrow-21.0.0/cpp/gdb_arrow.py \
    -O ${CONDA_PREFIX}/share/arrow/gdb/gdb_arrow.py

# 11. Handle cases:
# - If rootless docker: we want to run as root, which get's mapped to host user
# - If rootful docker: we don't want to run as root, but we want host user to access files (volum in docker-compose). So we change at runtime w/ Makefile (see $USER_FLAG).
# Create a non-root 'devuser' in the container for rootful docker/devcontainers, w/ sudo rights just in case you need to install apt packages later.
ARG USERNAME=devuser
RUN useradd -m -s /bin/bash $USERNAME && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 12. Give devuser ownership of mutable directories:
# - /home/faiss: so Faiss source can be patched and rebuilt without sudo
# - $CONDA_PREFIX: so `ninja install` and `pip install` work without sudo
# This eliminates all root/devuser permission issues during development.
RUN chown -R ${USERNAME}:${USERNAME} /home/faiss ${CONDA_PREFIX}

WORKDIR /home/Maximus

# root by default in the image e.g. in Rootless docker you want to be root to get auto-mapping to host user
USER root

CMD ["bash"]
