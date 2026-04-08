#include <stdio.h>

__global__ void add(int *A, int *B, int *C, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

int main() {
    int n = 1024;
    int size = n * sizeof(int);

    int h_A[1024], h_B[1024], h_C[1024];

    for (int i = 0; i < n; i++) {
        h_A[i] = i;
        h_B[i] = i;
    }

    int *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    add<<<4, 256>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Done Matrix Addition\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}