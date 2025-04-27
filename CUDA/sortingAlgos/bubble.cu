#include <stdio.h>
#include <cuda.h>

#define N 10

__global__ void bubbleSort(int *d_arr) {
    int tid = threadIdx.x;

    for (int i = 0; i < N; i++) {
        if (tid < N - i - 1) {
            if (d_arr[tid] > d_arr[tid + 1]) {
                int temp = d_arr[tid];
                d_arr[tid] = d_arr[tid + 1];
                d_arr[tid + 1] = temp;
            }
        }
        __syncthreads();  // Synchronize after every pass
    }
}

int main() {
    int h_arr[N] = {5, 9, 2, 7, 1, 3, 8, 6, 0, 4};
    int *d_arr;

    cudaMalloc((void**)&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    bubbleSort<<<1, N>>>(d_arr);
    cudaDeviceSynchronize();

    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sorted array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    cudaFree(d_arr);
    return 0;
}

