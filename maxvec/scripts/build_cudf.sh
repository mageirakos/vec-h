#!/bin/bash

export version="v25.08.00"

git clone https://github.com/rapidsai/cudf.git cudf_${version}
cd cudf_${version}

# # checkout this release:
git checkout tags/${version}

base_path="$HOME"

export CC=`which gcc`
export CXX=`which g++`
export CUDACXX=`which nvcc`

PARALLEL_LEVEL="$(nproc)" INSTALL_PREFIX=${base_path}/cudf_${version}_install ./build.sh libcudf --disable_nvtx --ptds

