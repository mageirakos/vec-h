#!/bin/bash
base_path="$HOME"
CC=gcc 
CXX=g++
export CUDACXX=/usr/local/cuda/bin/nvcc

echo "Using CC=$CC ($(which $CC), $($CC --version | head -n 1))"
echo "Using CXX=$CXX ($(which $CXX), $($CXX --version | head -n 1))" # 

OPENMP_VERSION=$($CXX -fopenmp -dM -E - < /dev/null 2>/dev/null | grep _OPENMP | awk '{print $3}')
echo "Using OpenMP version $OPENMP_VERSION (Should be > 200203 for at least OpenMP 2.0)"

# Downloading AMD BLIS/LibFlame BLAS/LAPACK
wget https://download.amd.com/developer/eula/blis/blis-4-0/aocl-blis-linux-gcc-4.0.tar.gz
wget https://download.amd.com/developer/eula/libflame/libflame-4-0/aocl-libflame-linux-gcc-4.0.tar.gz
tar -xzf aocl-blis-linux-gcc-4.0.tar.gz
tar -xzf aocl-libflame-linux-gcc-4.0.tar.gz
#ensure the folders exist
mkdir $base_path/include
mkdir $base_path/lib
cp -r amd-blis/include/* amd-libflame/include/* $base_path/include 
cp -r amd-blis/lib/* amd-libflame/lib/* $base_path/lib 
rm -r amd-blis amd-libflame

# Installing Rapids Raft (needed by CUVS)
git clone https://github.com/rapidsai/raft.git --branch v25.08.00
cd raft
bash ./build.sh libraft --cmake-args=\"-DCMAKE_INSTALL_PREFIX=$base_path\"
cd ../

# Installing Nvidia CUVS
git clone https://github.com/rapidsai/cuvs.git --branch v25.08.00
cd cuvs
# we disable the build tests for cuvs, as their require NCCL
bash ./build.sh libcuvs --no-mg --gpu-arch="86;90" --cmake-args=\"-DCMAKE_INSTALL_PREFIX=$base_path -DBUILD_TESTS=OFF\"
cd ../

# FAISS
git clone https://github.com/facebookresearch/faiss.git --branch v1.13.0
cd faiss
cmake_options=(
  "-DFAISS_ENABLE_GPU=ON"
  "-DFAISS_ENABLE_PYTHON=OFF"
  "-DFAISS_ENABLE_CUVS=ON"
  "-DCMAKE_BUILD_TYPE=Release"
  "-DFAISS_OPT_LEVEL=avx2"
  "-DBLA_VENDOR=AOCL"
  "-DBLAS_LIBRARIES=$base_path/lib/LP64/libblis-mt.a"
  "-DLAPACK_LIBRARIES=$base_path/lib/LP64/libflame.a;$base_path/lib/LP64/libblis-mt.a;-lpthread;-lm"
  "-DCMAKE_INCLUDE_PATH=$base_path/include"
  "-DCMAKE_LIBRARY_PATH=$base_path/lib"
  "-DCUDAToolkit_ROOT=/usr/local/cuda"
  "-DCMAKE_CUDA_ARCHITECTURES=native"
  "-DCMAKE_PREFIX_PATH=$base_path"
  "-DCMAKE_INSTALL_PREFIX=$base_path"
)
cmake -B build . "${cmake_options[@]}"
make -C build -j32
make -C build install