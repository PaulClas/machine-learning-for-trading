#!/usr/bin/env bash

## LLVM
sudo mkdir -p /opt/llvm/LLVM_3.8.0;
cd /opt/llvm/LLVM_3.8.0;
sudo wget http://releases.llvm.org/3.8.0/llvm-3.8.0.src.tar.xz;
sudo tar -xf llvm-3.8.0.src.tar.xz;
sudo mkdir build_llvm; cd build_llvm;
sudo cmake3 -G "Unix Makefiles" ../llvm-3.8.0.src
sudo make -j32
cd ..


sudo wget http://releases.llvm.org/3.8.0/cfe-3.8.0.src.tar.xz
sudo tar -xf cfe-3.8.0.src.tar.xz
sudo mkdir build_clang; cd build_clang
sudo cmake -G "Unix Makefiles" ../cfe-3.8.0.src -DLLVM_CONFIG=/opt/llvm/LLVM_3.8.0/build_llvm/bin/llvm-config
sudo make -j32


sudo -i;
mkdir -p /opt/catboost; cd /opt/catboost;
git clone https://github.com/catboost/catboost.git;
cd catboost/python-package/catboost;


../../../ya make -r -v -DUSE_ARCADIA_PYTHON=no -DOS_SDK=local -DPYTHON_CONFIG=/opt/intel/intelpython35/bin/python3-config -DCUDA_ROOT=/usr/local/cuda -DPYTHON_INCLUDE="-I/opt/intel/intelpython35/include/python3.5m" -DPYTHON_LIBRARIES=/opt/intel/intelpython35/lib -DPYTHON_BIN=/opt/intel/intelpython35/bin -DCUDA_NVCC_FLAGS="--compiler-bindir /opt/llvm/LLVM_3.8.0/build_clang/bin/clang++"


../../../ya make -r -v -DUSE_ARCADIA_PYTHON=no -DOS_SDK=local -DPYTHON_CONFIG=/home/stefan/.pyenv/versions/3.6.4/bin/python3-config -DCUDA_ROOT=/usr/local/cuda-9.0/ -DPYTHON_LIBRARIES=/home/stefan/.pyenv/versions/3.6.4/lib -DPYTHON_BIN=/home/stefan/.pyenv/versions/3.6.4/bin -DCUDA_NVCC_FLAGS="--compiler-bindir /opt/llvm/LLVM_3.8.0/build_clang/bin/clang++"



