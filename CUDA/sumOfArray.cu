#include<stdio.h>
#include<cuda.h>

#define N 10
#define BLOCK_SIZE 10

__global__ void sum(int* arr, int* sum){
        int idx = blockIdx.x*blockDim.x + threadIdx.x;

        if(idx<N){
                atomicAdd(sum, arr[idx]);
        }
}

int main(){
        int *h_arr, *d_arr, *d_sum, h_sum;

        size_t size = N*sizeof(int);

        h_arr = (int*) malloc(size);

        for(int i=0; i<N; i++){
                h_arr[i] = i;
        }

        cudaMalloc((void**)&d_arr, size);
        cudaMalloc((void**)&d_sum, sizeof(int));

        cudaMemset(d_sum, 0, sizeof(int));

        cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

        int blockSize = 1
        int gridSize = 1;

        sum<<<gridSize,blockSize>>>(d_arr, d_sum);

        cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

        printf("%d", h_sum);

        cudaFree(d_arr);
        cudaFree(d_sum);

        free(h_arr);

        return 0;
}

