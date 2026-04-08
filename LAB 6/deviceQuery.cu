#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int count;
    cudaGetDeviceCount(&count);

    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("GPU: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
        printf("Block Dim: %d %d %d\n",
               prop.maxThreadsDim[0],
               prop.maxThreadsDim[1],
               prop.maxThreadsDim[2]);
        printf("Grid Dim: %d %d %d\n",
               prop.maxGridSize[0],
               prop.maxGridSize[1],
               prop.maxGridSize[2]);
        printf("Global Memory: %llu\n", prop.totalGlobalMem);
        printf("Shared Memory: %llu\n", prop.sharedMemPerBlock);
        printf("Constant Memory: %llu\n", prop.totalConstMem);
        printf("Warp Size: %d\n", prop.warpSize);
    }
}