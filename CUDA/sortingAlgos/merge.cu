#include <stdio.h>
#include <cuda.h>

#define N 8

// Merge two sorted subarrays
__device__ void merge(int *arr, int left, int mid, int right) {
    int sizeLeft = mid - left + 1;
    int sizeRight = right - mid;

    int *leftArr = new int[sizeLeft];
    int *rightArr = new int[sizeRight];

    // Copy data to temporary arrays
    for (int i = 0; i < sizeLeft; i++)
        leftArr[i] = arr[left + i];
    for (int i = 0; i < sizeRight; i++)
        rightArr[i] = arr[mid + 1 + i];

    // Merge the temporary arrays
    int i = 0, j = 0, k = left;
    while (i < sizeLeft && j < sizeRight) {
        if (leftArr[i] <= rightArr[j]) {
            arr[k] = leftArr[i];
            i++;
        } else {
            arr[k] = rightArr[j];
            j++;
        }
        k++;
    }

    // Copy the remaining elements of leftArr[] and rightArr[], if any
    while (i < sizeLeft) {
        arr[k] = leftArr[i];
        i++;
        k++;
    }
    while (j < sizeRight) {
        arr[k] = rightArr[j];
        j++;
        k++;
    }

    delete[] leftArr;
    delete[] rightArr;
}

// Merge sort kernel
__global__ void mergeSort(int *arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        // Sorting left and right subarrays in parallel
        mergeSort<<<1, 1>>>(arr, left, mid);
        mergeSort<<<1, 1>>>(arr, mid + 1, right);

        // Merge the sorted halves
        merge(arr, left, mid, right);
    }
}

int main() {
    int h_arr[N] = {12, 11, 13, 5, 6, 7, 19, 10};
    int *d_arr;

    cudaMalloc((void**)&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    mergeSort<<<1, 1>>>(d_arr, 0, N - 1);
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

