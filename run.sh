mkdir build
cd build
cmake ../ -DUSE_QL=ON
# 或者cmake ../ -DUSE_CUDA=ON -DUSER_CUDA_ARCH=sm_90a
make
