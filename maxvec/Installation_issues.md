## Installation Issues

### Arrow
Run the following commands:
```bash
sudo apt install libboost-all-dev
sudo apt install libsnappy-dev
sudo apt install libbrotli-dev
sudo apt install thrift-compiler libthrift-dev
git clone https://github.com/Tencent/rapidjson.git
cd rapidjson
mkdir build && cd build
cmake ..
sudo make install
sudo apt install libre2-dev
```
Add the following to `cmake_options` of [`build_arrow.sh`](./scripts/build_arrow.sh):
```
   "-DRE2_INCLUDE_DIR=/usr/include/re2"
   "-DRE2_LIB=/usr/lib/x86_64-linux-gnu/libre2.so"
```

Do the following export after building Arrow and before building maximus:

```bash
export Parquet_DIR=~/arrow_install/lib/cmake
export ArrowAcero_DIR=~/arrow_install/lib/cmake
export Arrow_DIR=~/arrow_install/lib/cmake
export ArrowDataset_DIR=~/arrow_install/lib/cmake
```
