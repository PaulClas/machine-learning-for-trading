#!/usr/bin/env bash

sudo apt install nvidia-cuda-toolkit libcupti-dev gcc-6 python3-numpy python3-dev python3-pip python3-wheel;


# correct links
sudo mkdir -p /usr/local/cuda /usr/local/cuda/extras/CUPTI /usr/local/cuda/nvvm
sudo ln -s /usr/bin /usr/local/cuda/bin
sudo ln -s /usr/include /usr/local/cuda/include
sudo ln -s /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64
sudo ln -s /usr/local/cuda/lib64 /usr/local/cuda/lib
sudo ln -s /usr/include /usr/local/cuda/extras/CUPTI/include
sudo ln -s /usr/lib/x86_64-linux-gnu /usr/local/cuda/extras/CUPTI/lib64
sudo ln -s /usr/lib/nvidia-cuda-toolkit/libdevice /usr/local/cuda/nvvm/libdevice