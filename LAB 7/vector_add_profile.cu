#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

// CUDA kernel
__global__ void vecAdd(float *A, float *B, float *C) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < N)
        C[i] = A[i] + B[i];
}

int main() {
    float A[N], B[N], C[N];

    // Initialize data
    for(int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    float *dA, *dB, *dC;

    // Allocate GPU memory
    cudaMalloc((void**)&dA, N * sizeof(float));
    cudaMalloc((void**)&dB, N * sizeof(float));
    cudaMalloc((void**)&dC, N * sizeof(float));

    // Copy to device
    cudaMemcpy(dA, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch kernel
    vecAdd<<<(N + 255) / 256, 256>>>(dA, dB, dC);

    cudaDeviceSynchronize();  // ensure kernel finished

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy result back
    cudaMemcpy(C, dC, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Kernel Time: %f ms\n", ms);

    // 🔥 Measured Bandwidth
    float totalBytes = 3 * N * sizeof(float);  // read A + read B + write C
    float timeSec = ms / 1000.0f;

    float measuredBW = totalBytes / timeSec / 1e9;
    printf("Measured Bandwidth: %f GB/s\n", measuredBW);

    // 🔥 Device Info (safe fields only)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("\nDevice Info:\n");
    printf("GPU Name: %s\n", prop.name);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);

    // 🔥 Theoretical Bandwidth (explained, not computed from missing fields)
    printf("\nTheoretical Bandwidth: Not directly computed (missing fields in this setup)\n");

    // Cleanup
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}