#!/bin/bash

# Clone the repository
git clone --recursive https://github.com/taskflow/taskflow --branch v3.11.0
cd taskflow

# Define compilers and build types
build_type="Release"

base_path="$HOME"

# Create build folder
build_folder="build"
mkdir "$build_folder"
cd "$build_folder"

# Set CMake options and run CMake
cmake_options=(
  "-DCMAKE_BUILD_TYPE=$build_type"
  "-DCMAKE_INSTALL_PREFIX=$base_path/taskflow_install"
  "-DTASKFLOW_BUILD_TESTS=OFF"
  "-GNinja"
)

# Run CMake with specified options
CC=gcc CXX=g++ cmake "${cmake_options[@]}" ..

# Build and install
ninja -j 8
ninja install

# Go back to the parent directory
cd ..

