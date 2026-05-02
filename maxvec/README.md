<!---
[![pipeline status](https://gitlab.inf.ethz.ch/PUB-SYSTEMS/eth-dataprocessing/Maximus/badges/main/pipeline.svg)](https://gitlab.inf.ethz.ch/PUB-SYSTEMS/eth-dataprocessing/Maximus/-/commits/main)
--->

<div>
<centering>
<p align="center"><img src="./assets/maximus-logo.svg" width="70%"></p>
</centering>
</div>

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Benchmarking](#benchmarking)
- [Testing](#testing)

## Overview

Maximus is a modular, accelerated query engine for data analytics (OLAP) on Heterogeneous Systems, developed in the Systems Group at ETH Zurich (Swiss Federal Institute of Technology).
Through the concept of operator-level integration, Maximus can use operators from third-party engines and achieve even better performance with these operators than when they are used within their native engines.
In the current version, Maximus integrates operators from Apache Acero (CPU) and cuDF (GPU), but its modular design and flexibility allow it to easily integrate operators from other engines, as well.
Maximus supports all TPC-H queries on both the CPU and the GPU and can achieve performance comparable to that of the best systems available but with a far higher degree of completeness and flexibility.

## Features

Maximus has the following features:
- modular and flexible design
- operator-level integration:
  - integrates CPU operators from Apache Acero
  - integrates GPU operators from cuDF
- push-based query execution
- uses columnar storage (arrow format)
- accelerators support (GPUs)

## Dependencies

The installation requires the following dependencies:
- C/C++ compiler (tested with clang and gcc): tested with GCC `v11.4.0` and `CLANG v17.0.0`.
- Apache Arrow, which can be installed using [this script](./scripts/build_arrow.sh). Since Maximus `v0.3`, Apache Arrow `v21.0.0` is required. If you had issues in installing arrow see [here](./Installation_issues.md)
- Taskflow, which can be installed using [this script](./scripts/build_taskflow.sh).
- cuDF (optional, only used in a GPU version), which can be installed using [this script](./scripts/build_cudf.sh). Since Maximus `v0.3`, cuDF `v25.08.00` is required.
- Caliper Profiler (optional, only used when profiling enabled), which can be installed using [this script](./scripts/build_caliper.sh).
- Faiss for vector similiary search (optional), which can be installed using [this script](./scripts/build_faiss.sh).

## Installation

To build and install Maximus, do the following steps:
```bash
###############
# get Maximus
###############
git clone https://gitlab.inf.ethz.ch/PUB-SYSTEMS/eth-dataprocessing/Maximus maximus && cd maximus

##############################
# build and install Maximus
##############################
mkdir build && cd build

# set up the compiler you want to use, e.g. with:
export CC=`which cc`
export CXX=`which CC`

# run cmake with optional parameters
cmake -DCMAKE_BUILD_TYPE=Release -DMAXIMUS_WITH_TESTS=ON -DCMAKE_PREFIX_PATH=<path where dependencies installed> ..

# compile
make -j 8

# install
make install
```

The overview of available CMake options is given in the table below. The default values are given in **bold**. These options can be passed in the `cmake` command with the prefix `-D`, as in `cmake -DCMAKE_BUILD_TYPE=Release`.

CMAKE OPTION | POSSIBLE VALUES            | DESCRIPTION
| :------------------- |:---------------------------|:------------------- |
`CMAKE_BUILD_TYPE` | **Release**, Debug         | Whether to build in the release or debug mode.
`CMAKE_INSTALL_PREFIX` | any path, by default `./`  | The path where to install Maximus.
`CMAKE_PREFIX_PATH` | any path, defined by CMake | The path(s) where to look for dependencies
`BUILD_SHARED_LIBS` | **ON**, OFF                | Whether to build Maximus as a shared library (ON) or as a static library (OFF).
`MAXIMUS_WITH_TESTS` | **ON**, OFF                | Whether to build the tests as well.
`MAXIMUS_WITH_BENCHMARKS` | **ON**, OFF                | Whether to build the benchmarks as well.
`MAXIMUS_WITH_GPU` | ON, **OFF**                | Whether to build the GPU backend.
`MAXIMUS_WITH_PROFILING` | ON, **OFF**                | Whether to enable profiling.
`MAXIMUS_WITH_SANITIZERS` | ON, **OFF**                | Enable the sanitizers for all the targets.
`MAXIMUS_WITH_DATASET_API` | **ON**, OFF              | Build Maximus with the Dataset API support.
`MAXIMUS_WITH_FAISS` | CPU, GPUCUVS, **OFF** | Wether to enable Faiss integration.

### Option 2: Installation with Docker (Rootless/Rootful Compatible)

This setup is best used for development, and it uses Docker with `docker-compose` to create a reproducible, isolated build environment with all necessary dependencies (CUDA, Mamba, Faiss, custom Arrow, etc.).
1. `$ make setup OPT_LEVEL=[avx512,avx2,sve,generic] PARALLEL_JOBS=[..]`: Builds the Docker image, starts the container, and does an initial Maximus build in `./build`.  
   - defaults (auto): OPT_LEVEL=avx512/sve (x86_64/aarch64), PARALLEL_JOBS=8
2. `$ make shell`: Use this to open shell, and then run any subsequent commands. This is safe and does not run as root user. Whereas `$docker exec -it maxvs-dev-gpu bash` is unsafe and runs as root.
    - `$make is_rootless`: To check if you are running rootless docker or not

**File Permissions & Host Access (Rootless/Rootful):**
To avoid permission issues on the host filesystem (which is volume-mounted) we would like to ensure all container commands run as host user ID (`UID:GID`). We change the user at runtime which guarantees that files created inside the container (e.g., in the `./build` directory) are owned by host user on the host machine.
- The `Makefile` dynamically detects if the Docker daemon is running in **Rootless** mode.
    - If **Rootless**: the container's `root` user is automatically mapped to your host user by the daemon. So we don't do any runtime changes and run as `root` inside the container. 
    - If **Rootful** (standard Docker): the container starts as `root`, but all build/run commands (`make shell`, `make build-maximus`, etc.) use the argument `--user $(UID):$(GID)`. This switches the process to run directly as your host user.

Available commands in `Makefile`:

| Command | Description |
| :--- | :--- |
| `$ make is_rootless` | Check if you are running rootless docker or not. |
| `$ make shell` | Opens an interactive Bash shell inside the container. |
| `$ make test` | Runs the full test suite (`ninja test`) inside the container. |
| `$ make build-maximus` | Executes CMake configuration and Ninja compilation of Maximus inside the container. |
| `$ make reset` | Cleans the local `./build` directory and fully rebuilds Maximus. |
| `$ make logs` | Displays and follows the real-time output and logs of the running container. |
| `$ make build-image` | Builds the Docker image. |
| `$ make compose-up` | Starts the container in detached mode (`-d`). |
| `$ make compose-down` | Stops and removes the running container and its network. |


## Benchmarking

Running the full TPC-H benchmark is possible by running the `maxbench` executable. Within the `build` folder, it can be run e.g. with:
```bash
./benchmarks/maxbench --benchmark=tpch --queries=q1,q2,q3 --device=cpu --storage_device=cpu --engines=maximus --n_reps=5 --path=<path to CSV files>
```
Possible values for each of these options are:
- `benchmark`: the benchmarking suite to you, can be one of `tpch`, `clickbench` or `h2o`.
- `queries`: any comma-separated list of queries. For example, for the tpch benchmark, any subset of `q1, q2, ... , q22` is valid.
- `device`: `cpu`, `gpu`. This describes where the query will be executed.
- `storage_device`: `cpu`, `cpu-pinned`, `gpu`. This describes where the tables are initially stored. The cpu-pinned memory requires the GPU-backend.
- `engines`: `maximus`, `acero`. This specifies which engine to use. Only the `maximus` options supports the GPU backend in this benchmark.
- `path`: any path where the tables are located.
- `csv_batch_size`: a positive integer, or in a form of a power e.g. `2^20`. specifying the batch size of the tables that are initially stored on the CPU.
- `n_reps`: the number of repetitions.

For a full list of all available command-line arguments, use:
```bash
./benchmarks/maxbench --help
```
Running this e.g. for a query q1, will yield the following output:
```bash
===================================
          LOADING TABLES
===================================
Loading tables to:                   cpu
Loading times over repetitions [ms]: 25,
===================================
    MAXBENCH TPCH BENCHMARK:    
===================================
---> benchmark:                TPCH
---> queries:                  q1
---> Tables path:              /../maximus/tests/tpch/csv-0.01
---> Engines:                  maximus
---> Number of reps:           1
---> Device:                   cpu
---> Storage Device:           cpu
---> Num. outer threads:       1
---> Num. inner threads:       8
---> Operators Fusion:         OFF
---> CSV Batch Size (string):  2^30
---> CSV Batch (number):       1073741824
---> Tables initially pinned:  NO
---> Tables as single chunk:   NO
===================================
            QUERY q1
===================================
---> TPC-H query: q1
---> Query Plan:
NATIVE::QUERY PLAN ROOT
    └── NATIVE::TABLE SINK
        └── ACERO::ORDER BY
            └── ACERO::GROUP BY
                └── ACERO::PROJECT
                    └── ACERO::FILTER
                        └── NATIVE::TABLE SOURCE

Query Plan after scheduling:
NATIVE::QUERY PLAN ROOT
    └── NATIVE::TABLE SINK
        └── ACERO::ORDER BY
            └── ACERO::GROUP BY
                └── ACERO::PROJECT
                    └── ACERO::FILTER
                        └── NATIVE::TABLE SOURCE

===================================
              RESULTS
===================================
maximus RESULTS (top 10 rows):
l_returnflag(utf8),l_linestatus(utf8),sum_qty(double),sum_base_price(double),sum_disc_price(double),sum_charge(double),avg_qty(double),avg_price(double),avg_disc(double),count_order(int64)
A,F,380456.000000,532348211.649999,505822441.486100,526165934.000840,25.575155,35785.709307,0.050081,14876
N,F,8971.000000,12384801.370000,11798257.208000,12282485.056933,25.778736,35588.509684,0.047759,348
N,O,742847.000000,1041580998.800000,989812549.690601,1029499565.063830,25.455658,35692.584429,0.049931,29182
R,F,381449.000000,534594445.350000,507996454.406700,528524219.358904,25.597168,35874.006533,0.049828,14902


===================================
              TIMINGS
===================================
Execution times [ms]:
- MAXIMUS TIMINGS [ms]: 	2
Execution stats (min, max, avg):
- MAXIMUS STATS: MIN = 2 ms; 	MAX = 2 ms; 	AVG = 2 ms
===================================
        SUMMARIZED TIMINGS
===================================
--->Results saved to ./results.csv
cpu,maximus,q1,2,
```

## Testing

To run the unit tests, run the following inside the build folder:
```bash
make test
```
This will run all the tests and report for each of them if it failed or not. However, each of these tests might include multiple smaller tests. To get the overview of all the tests, run:
```bash
ctest --verbose make check
```

If some of the tests have failed, the output can be seen by running `vim ./Testing/Temporary/LastTest.log` inside the `build` folder.

Alternatively, a specific test can be run, e.g. by:
```bash
./tests/test.tpch
```
