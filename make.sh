#!/usr/bin/env bash
echo "Making serial.cpp"
g++ -std=c++11 -g -o serial shared.cpp serial.cpp
echo "Making parallel.cu"
nvcc -std=c++11 -g -G -o parallel shared.cpp parallel.cu
