#!/usr/bin/env bash

sudo apt install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev -y;

# requires gcc 6
sudo apt install gcc-6 g++-6 -y;

# latest installer: https://developer.nvidia.com/cuda-downloads
# toolkit archive: https://developer.nvidia.com/cuda-toolkit-archive
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run


sudo sh cuda-9.0.run --silent --toolkit --toolkitpath=/usr/local/cuda-9.0