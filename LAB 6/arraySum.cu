#include <stdio.h>
#include <cuda_runtime.h>

__global__ void sumKernel(float *a, float *result, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n) {
        atomicAdd(result, a[i]);
    }
}

int main() {
    int n = 1024;
    float h_a[1024];
    float h_result = 0.0f;

    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
    }

    float *d_a, *d_result;

    cudaMalloc((void**)&d_a, n * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));

    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);

    // initialize properly
    cudaMemset(d_result, 0, sizeof(float));

    sumKernel<<<4, 256>>>(d_a, d_result, n);

    // 🔥 CHECK FOR ERRORS
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sum = %f\n", h_result);

    cudaFree(d_a);
    cudaFree(d_result);

    return 0;
}