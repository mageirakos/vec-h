################################################################################
#                                                                              #
#  Dockerfile for the BigVectorBench benchmarks with Maximus&Faiss 08/2025     #
#                                                                              #
#  Description:                                                                #
#    - This is primarely intended to make the benchmarking environment         #
#      reproducible.                                                           #   
#    - Maintainability is not the first priority                               #
#                                                                              #                     
#   Intended Target Machine Specs:                                             #
#    - AMD EPYC 9124 3GHz. (AVX512)                                            #
#    - H100                (arch=90)                                           #
#                                                                              #
#  Usage:                                                                      #
#    docker build -f /path/to/Dockerfile -t maximus-dev /path/to/Maximus       #
#    docker run --gpus all -it maximus-dev bash                                #
#    cmake -S . -B build -DMAXIMUS_WITH_GPU=ON -DMAXIMUS_WITH_FAISS=GPUCUVS \  #
#           -G Ninja -DCMAKE_PREFIX_PATH="/usr/local"                          #
#    cmake --build build -- -j12                                               #
#    cd build && ninja test                                                    #           
#  Note: Check that you link against faiss_avx512                              #
#                                                                              #
################################################################################
FROM nvidia/cuda:12.5.1-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /home
ENV INSTALL_DIR=/usr/local
ENV BUILD_PROC=8
ENV BUILD_CUDA_ARCH=90
ENV BUILD_FAISS_OPT_LEVEL=avx512
ENV BUILD_BLIS_ISA=AVX512

# Install essentials
RUN apt-get update && apt-get install -y\
    build-essential \
    clang \
    cmake \
    ninja-build \
    git \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install CMake from official binaries
ARG CMAKE_VERSION=3.31.8
RUN wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && \
    tar -xzf cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz --strip-components=1 -C /usr/local && \
    rm cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz

# Install further packages
RUN apt-get update && apt-get install -y \
    libboost-all-dev \
    libgtest-dev \
    libgmock-dev \
    liblz4-tool \
    protobuf-compiler \
    libprotobuf-dev \
    rapidjson-dev \
    libsnappy-dev \
    libthrift-dev \
    zlib1g-dev \
    libzstd-dev \
    libblas-dev \
    liblapack-dev \
    libomp-dev \
    libgflags-dev \
    libnccl2 \
    libnccl-dev \
    libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# 1. Install Taskflow
RUN git clone --recursive https://github.com/taskflow/taskflow && \
    cd taskflow && \
    cmake -S . -B build -G Ninja \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} && \
    cmake --build build -j${BUILD_PROC} && \
    cmake --build build --target install && \
    cd .. && rm -rf taskflow

# 2. Install Arrow
RUN git clone --recursive https://github.com/apache/arrow --branch apache-arrow-19.0.0 && \
    cd arrow/cpp && \
    mkdir build && cd build && \
    CC=gcc CXX=g++ cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
      -GNinja \
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
      -DARROW_JEMALLOC=ON \
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
      -DARROW_CXXFLAGS=-w \
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
      -DProtobuf_SOURCE=BUNDLED && \
    ninja -j${BUILD_PROC} && \
    ninja install && \
    cd ../.. && rm -rf arrow

# 3. Install CUDF like in build_cudf.sh
ENV PARALLEL_LEVEL=${BUILD_PROC}
RUN git clone https://github.com/rapidsai/cudf.git cudf && \
    cd cudf && \
    git checkout tags/v25.06.00a && \
    CC=gcc CXX=g++ CUDACXX=nvcc INSTALL_PREFIX=${INSTALL_DIR} \
    ./build.sh libcudf --disable_nvtx --ptds \
    --cmake-args=\"-DCMAKE_PREFIX_PATH=${INSTALL_DIR} -Drapids-cmake-version=25.06\" && \
    cd .. && rm -rf cudf

# 4. Install Faiss like in build_faiss.sh
RUN git clone https://github.com/amd/blis && \
    cd blis && \
    ./configure --prefix=${INSTALL_DIR} \
                --enable-cblas \
                --enable-shared \
                --enable-threading=openmp \
                CC=gcc CFLAGS="-O3 -march=native" CXXFLAGS="-O3 -march=native" auto && \
    make -j${BUILD_PROC} && \
    make install && \
    cd .. && rm -rf blis

RUN git clone https://github.com/rapidsai/raft.git --branch v25.06.00 && \
    cd raft && \
    bash ./build.sh libraft \
         --cmake-args=\"-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}\" && \
    cd .. && rm -rf raft

RUN git clone https://github.com/rapidsai/cuvs.git --branch v25.06.00 && \
    cd cuvs && \
    bash ./build.sh libcuvs \
         --no-mg \
         --gpu-arch="${BUILD_CUDA_ARCH}" \
         --cmake-args=\"-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}\" && \
    cd .. && rm -rf cuvs

RUN git clone https://github.com/amd/aocl-utils.git && \
    cd aocl-utils && \
    cmake -B default -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} && \
    cmake --build default --config release -j${BUILD_PROC} && \
    cmake --install default --config release && \
    cd .. && rm -rf aocl-utils

ENV AOCL_ROOT=${INSTALL_DIR}
RUN git clone https://github.com/amd/libflame.git && \
    cd libflame && \
    mkdir newbuild && cd newbuild && \
    cmake .. -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
             -DENABLE_AMD_AOCC_FLAGS=ON \
             -DENABLE_AOCL_BLAS=ON \
             -DLF_ISA_CONFIG=${BUILD_BLIS_ISA} && \
    make -j${BUILD_PROC} && \
    make install && \
    cd ../.. && rm -rf libflame

RUN git clone https://github.com/facebookresearch/faiss.git --branch v1.11.0 && \
    cd faiss && \
    cmake -B build . \
        -DFAISS_ENABLE_GPU=ON \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DFAISS_ENABLE_CUVS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DFAISS_OPT_LEVEL=${BUILD_FAISS_OPT_LEVEL} \
        -DBLA_VENDOR=AOCL \
        -DBLAS_LIBRARIES="${INSTALL_DIR}/lib/libblis-mt.so" \
        -DLAPACK_LIBRARIES="${INSTALL_DIR}/lib/libflame.so;${INSTALL_DIR}/lib/libblis-mt.so;-lpthread;-lm" \
        -DCMAKE_INCLUDE_PATH="${INSTALL_DIR}/include" \
        -DCMAKE_LIBRARY_PATH="${INSTALL_DIR}/lib" \
        -DCMAKE_CUDA_ARCHITECTURES=${BUILD_CUDA_ARCH} \
        -DCMAKE_PREFIX_PATH="${INSTALL_DIR}" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" && \
    make -C build -j${BUILD_PROC} && \
    make -C build install && \
    cd .. && rm -rf faiss

# 5. Install Caliper
RUN git clone https://github.com/LLNL/Caliper.git caliper && \
    cd caliper && \
    build_folder="build" && \
    mkdir "$build_folder" && \
    cd "$build_folder" && \
    cmake -GNinja .. \
      -DCMAKE_BUILD_TYPE="Release" \
      -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" && \
    ninja -j${BUILD_PROC} && \
    ninja install && \
    cd ../.. && rm -rf caliper

# Installing Maximus
COPY . /home/Maximus
WORKDIR /home/Maximus
RUN echo "Building Maximus" && \
    git reset --hard && git clean -fdx && \
    git checkout -f feature/vector-similarity-search && \
    git checkout -f 811cebc44ac46140fcb90c285550c91a8a356b70 && \
    rm -rf build && mkdir -p build && cd build && \
    cmake .. -DMAXIMUS_WITH_TESTS=ON \
             -DMAXIMUS_WITH_GPU=ON \
             -DMAXIMUS_WITH_FAISS=GPUCUVS \
             -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_PREFIX_PATH=${INSTALL_DIR} \
             -DCMAKE_CUDA_ARCHITECTURES=${BUILD_CUDA_ARCH} \
             -GNinja && \
    ninja -j${BUILD_PROC}

CMD ["bash"]