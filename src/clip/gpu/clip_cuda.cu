#include <cuda.h>
#include <cub/cub.cuh>

template<typename T>
__global__ void clipKernel(T const *input, T *output, float minValue, float maxValue, int n, int numThread){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = 0; i < numThread; i++){
        int index = tid * numThread + i;
        if(index >= n){
            break;
        }
        float tmp = (static_cast<float>(input[index]) < minValue ? minValue : (static_cast<float>(input[index]) > maxValue ? maxValue : static_cast<float>(input[index])));
        output[index] = static_cast<T>(tmp);
    }
}
template<typename T>
void clipLaunch(void const *input, void *output, float minValue, float maxValue, int n){
    int numThread = 2;
    int BLOCK_DIM = 512;
    int num_block = (n + numThread * BLOCK_DIM - 1) / (numThread * BLOCK_DIM);
    clipKernel<<<num_block, BLOCK_DIM>>>((T *)input, (T *)output, minValue, maxValue, n, numThread);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Possibly: exit(-1) if program cannot continue....
    }

}
extern "C" void clip_cuda(void const *input, void *output, float minValue, float maxValue, int n, int byteSize)
{
    if(byteSize == 2){
        clipLaunch<half>(input, output, minValue, maxValue, n);
    }
    else if(byteSize == 4){
        clipLaunch<float>(input, output, minValue, maxValue, n);
    }
}
