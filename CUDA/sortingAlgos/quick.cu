#include <stdio.h>
#include <cuda.h>

#define N 8

// Partition kernel
__device__ int partition(int *arr, int low, int high) {
    int pivot = arr[high];  // Pivot element is taken as the last element
    int i = low - 1;        // Pointer for the smaller element

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }

    // Swap pivot element to correct position
    int temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;

    return (i + 1);
}

// QuickSort kernel
__global__ void quickSort(int *arr, int low, int high) {
    if (low < high) {
        int pivotIndex = partition(arr, low, high);

        // Recur on left and right parts
        quickSort<<<1, 1>>>(arr, low, pivotIndex - 1);  // Left part
        quickSort<<<1, 1>>>(arr, pivotIndex + 1, high); // Right part
    }
}

int main() {
    int h_arr[N] = {12, 11, 13, 5, 6, 7, 19, 10};
    int *d_arr;

    // Allocate memory on device
    cudaMalloc((void**)&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch quickSort kernel
    quickSort<<<1, 1>>>(d_arr, 0, N - 1);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the sorted array
    printf("Sorted array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_arr);
    return 0;
}

