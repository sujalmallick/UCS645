#include <stdio.h>

#define N 1024

__global__ void iterativeSum(int *arr, int *result) {
    int sum = 0;
    for(int i = 0; i < N; i++)
        sum += arr[i];
    result[0] = sum;
}

__global__ void formulaSum(int *result) {
    int n = N;
    result[1] = (n * (n + 1)) / 2;
}

int main() {
    int h_arr[N];
    for(int i=0;i<N;i++) h_arr[i]=i+1;

    int h_result[2];

    int *d_arr, *d_result;
    cudaMalloc(&d_arr, N*sizeof(int));
    cudaMalloc(&d_result, 2*sizeof(int));

    cudaMemcpy(d_arr, h_arr, N*sizeof(int), cudaMemcpyHostToDevice);

    iterativeSum<<<1,1>>>(d_arr, d_result);
    formulaSum<<<1,1>>>(d_result);

    cudaMemcpy(h_result, d_result, 2*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Iterative Sum: %d\n", h_result[0]);
    printf("Formula Sum: %d\n", h_result[1]);

    cudaFree(d_arr);
    cudaFree(d_result);
}