#!/bin/bash

export version="21.0.0"

# Clone the repository
git clone --recursive https://github.com/apache/arrow --branch apache-arrow-${version} arrow_${version}
cd arrow_${version}/cpp

base_path="$HOME"

# Create build folder
build_folder="build"
mkdir "$build_folder"
cd "$build_folder"

export CC=`which gcc`
export CXX=`which g++`

# export cpu_flags="-mcpu=neoverse-v2+crypto+sha3+sm4+sve2-aes+sve2-sha3+sve2-sm4"

# Default CPU flags
cpu_flags="-mcpu=native"

# Check if soc_id exists (GH200 / Grace)
SOC_ID_FILE="/sys/devices/soc0/soc_id"
if [[ -f "$SOC_ID_FILE" ]]; then
    soc_id=$(cat $SOC_ID_FILE | tr -d '[:space:]')
    case "$soc_id" in
        "jep106:036b:0241")
            echo "Detected NVIDIA Grace Hopper (GH200)"
            cpu_flags="-mcpu=neoverse-v2+crypto+sha3+sm4+sve2-aes+sve2-sha3+sve2-sm4"
            ;;
        *)
            echo "Detected ARM SoC: $soc_id, using generic ARM flags"
            cpu_flags="-mcpu=native"
            ;;
    esac
else
    # Fallback: generic ARM detection
    arch=$(uname -m)
    if [[ "$arch" == "aarch64" ]]; then
        echo "Detected generic ARM64 system"
        cpu_flags="-mcpu=native"
    elif [[ "$arch" == "x86_64" ]]; then
        echo "Detected x86_64 system"
        cpu_flags="-march=native"
    else
        echo "Unknown architecture ($arch), using safe default"
        cpu_flags=""
    fi
fi

echo "Using CPU flags: $cpu_flags"

# Set CMake options and run CMake
cmake_options=(
        "-DCMAKE_C_FLAGS=${cpu_flags}"
        "-DCMAKE_CXX_FLAGS=${cpu_flags}"
	"-DCMAKE_BUILD_TYPE=Release"
	"-DCMAKE_INSTALL_PREFIX=$base_path/arrow_${version}_install"
	"-GNinja"
	"-Dxsimd_SOURCE=BUNDLED"
	"-DARROW_WITH_UTF8PROC=OFF"
	"-DARROW_WITH_RE2=ON"
	"-DARROW_FILESYSTEM=ON"
	"-DARROW_TESTING=OFF"
	"-DARROW_WITH_LZ4=OFF"
	"-DARROW_WITH_ZSTD=OFF"
	"-DARROW_BUILD_STATIC=ON"
	"-DARROW_BUILD_SHARED=ON"
	"-DARROW_BUILD_TESTS=OFF"
	"-DARROW_BUILD_BENCHMARKS=OFF"
	"-DARROW_IPC=ON"
	"-DARROW_FLIGHT=OFF"
	"-DARROW_COMPUTE=ON"
	"-DARROW_CUDA=OFF"
	"-DARROW_JEMALLOC=ON"
	"-DARROW_USE_GLOG=OFF"
	"-DARROW_DATASET=ON"
	"-DARROW_BUILD_UTILITIES=OFF"
	"-DARROW_HDFS=OFF"
	"-DCMAKE_VERBOSE_MAKEFILE=ON"
	"-DARROW_TENSORFLOW=OFF"
	"-DARROW_CSV=ON"
	"-DARROW_JSON=ON"
	"-DARROW_WITH_BROTLI=ON"
	"-DARROW_WITH_SNAPPY=ON"
	"-DARROW_WITH_ZLIB=ON"
	"-DARROW_PARQUET=ON"
	"-DARROW_SUBSTRAIT=OFF"
	"-DCMAKE_VERBOSE_MAKEFILE=ON"
	"-DARROW_ACERO=ON"
	"-DARROW_WITH_BACKTRACE=ON"
	"-DARROW_CXXFLAGS=-w"
	"-DARROW_S3=OFF"
	"-DARROW_ORC=OFF"
	"-DARROW_POSITION_INDEPENDENT_CODE=ON"
	"-DARROW_DEPENDENCY_USE_SHARED=ON"
	"-DARROW_BOOST_USE_SHARED=ON"
	"-DARROW_BROTLI_USE_SHARED=ON"
	"-DARROW_GFLAGS_USE_SHARED=ON"
	"-DARROW_GRPC_USE_SHARED=ON"
	"-DARROW_PROTOBUF_USE_SHARED=ON"
	"-DARROW_ZSTD_USE_SHARED=ON"
	"-DProtobuf_SOURCE=BUNDLED"
)

# Run CMake with specified options
cmake "${cmake_options[@]}" ..

# Build and install
ninja -j "$(nproc)"
ninja install

# Go back to the parent directory
cd ..
