CUDA C Setup Guide for Dell Precision 3660 (Windows 11/WSL)
1. GPU Verification & Driver Installation (Windows)

Check GPU Model:

Open Device Manager → Expand Display adapters
Confirm NVIDIA GeForce RTX GPU model
Visually verify "GEFORCE RTX" through PC vents


Install Drivers:

Download GeForce Game Ready Driver 572.70
Run installer with standard defaults



2. WSL Setup & CUDA Toolkit Installation
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-wsl-ubuntu-12-8-local_12.8.1-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-8-local_12.8.1-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
sudo apt-get upgrade

Environment Configuration:
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

3. Verification Tests
git clone https://github.com/NVIDIA/cuda-samples.git
sudo apt install cmake
cd cuda-samples
mkdir build && cd build
cmake .. && make -j$(nproc)

Run diagnostics:

Device query:
./deviceQuery


Bandwidth test:
./bandwidthTest



4. First CUDA Program
Create file:
vim RollNumberLab5.cu

Vim commands:

Press i to insert code
ESC → :write to save
ESC → :quit to exit

Compile & Execute:
nvcc RollNumberLab5.cu -o RollNumberLab5.out
./RollNumberLab5.out

