#include <stdio.h>
#include <stdlib.h>

#define N 1000

__global__ void simpleSort(int *arr) {
    int i = threadIdx.x;

    for(int j = i+1; j < N; j++) {
        if(arr[i] > arr[j]) {
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
}

int main() {
    int h_arr[N];

    for(int i=0;i<N;i++)
        h_arr[i] = rand() % 1000;

    int *d_arr;
    cudaMalloc(&d_arr, N*sizeof(int));

    cudaMemcpy(d_arr, h_arr, N*sizeof(int), cudaMemcpyHostToDevice);

    simpleSort<<<1, N>>>(d_arr);

    cudaMemcpy(h_arr, d_arr, N*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sorted first 10:\n");
    for(int i=0;i<10;i++)
        printf("%d ", h_arr[i]);

    cudaFree(d_arr);
}