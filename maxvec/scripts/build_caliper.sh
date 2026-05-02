#!/bin/bash

# Clone the repository
git clone https://github.com/LLNL/Caliper.git caliper
cd caliper

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
  "-DCMAKE_INSTALL_PREFIX=$HOME/caliper_install"
  "-GNinja"
  ".."
)

# Run CMake with specified options
cmake "${cmake_options[@]}"

# Build and install
ninja -j 8
ninja install

# Go back to the parent directory
cd ..

