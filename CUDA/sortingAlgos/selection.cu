#include <stdio.h>
#include <cuda.h>

#define N 10

// Kernel to perform selection sort
__global__ void selectionSort(int *d_arr) {
    int idx = threadIdx.x;

    for (int i = 0; i < N - 1; i++) {
        int minIdx = i;
        for (int j = i + 1; j < N; j++) {
            if (d_arr[j] < d_arr[minIdx]) {
                minIdx = j;
            }
        }

        // Swap the minimum element with the first unsorted element
        if (minIdx != i) {
            int temp = d_arr[i];
            d_arr[i] = d_arr[minIdx];
            d_arr[minIdx] = temp;
        }
    }
}

int main() {
    int h_arr[N] = {64, 25, 12, 22, 11, 90, 19, 3, 7, 1};
    int *d_arr;

    cudaMalloc((void**)&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    selectionSort<<<1, 1>>>(d_arr);
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

