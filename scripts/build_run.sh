#!/bin/bash

# remove old build directory
rm -rf build

# build
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo

# compile and install in parallel
cmake --build build -j $(sysctl -n hw.ncpu || nproc)

# install in build directory
cmake --install build --prefix build

# run the engine
./build/bin/pie_engine --model .models/llama3-8b-instruct --ipc "/pie_bulk_data"
