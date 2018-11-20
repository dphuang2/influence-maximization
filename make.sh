#!/usr/bin/env bash
echo "Making serial.cpp"
g++ -std=c++11 -g -o serial serial.cpp
echo "Making parallel.cu"
nvcc -std=c++11 -g -o parallel parallel.cu
