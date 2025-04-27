#include <cuda.h>
#include <stdio.h>

#define N 20000000

__global__ void add(int* a, int* b, int* c, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < n)
        {
                c[idx] = a[idx] + b[idx];
        }
}


void populate(int *h_a, int *h_b, int n){
        for(int i=0; i<n;i++)
        {
                h_a[i] = rand()%1000;
                h_b[i] = rand()%1000;
        }
}


int main(){
        int *h_a, *h_b, *h_c;
        int *d_a, *d_b, *d_c;

        size_t size = N*sizeof(int);

        h_a = (int*) malloc(size);
        h_b = (int*) malloc(size);
        h_c = (int*) malloc(size);

        populate(h_a, h_b, N);

        cudaMalloc((void**)&d_a, size);
        cudaMalloc((void**)&d_b, size);
        cudaMalloc((void**)&d_c, size);

        cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

        clock_t start = clock();
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;

        add<<<gridSize, blockSize>>> (d_a, d_b, d_c, N);

        cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

        for(int i=0; i<10; i++){
                printf("%d", h_c[i]);
        }

        printf("\n");

        clock_t stop = clock();

        double timetaken = (double) (stop-start)/CLOCKS_PER_SEC;
        printf("\nTime : %1f\n", timetaken);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        free(h_a);
        free(h_b);
        free(h_c);

        return 0;

}
