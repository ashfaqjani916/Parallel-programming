#include <cuda.h>
#include <stdio.h>

#define N 100

__global__ void mul(float* a, float* b, float* c, int width){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < width && col < width){
        float value = 0.0f;
        for(int i = 0; i < width; i++){
            value += a[row * width + i] * b[i * width + col];
        }
        c[row * width + col] = value;
    }
}

// Initialize matrix with random values
void init(float* a, int size){
    for(int i = 0; i < size; i++){
        a[i] = rand() % 100;
    }
}

int main(){
    int size = N * N * sizeof(float);

    // Allocate memory on host
    float *h_a = (float*) malloc(size);
    float *h_b = (float*) malloc(size);
    float *h_c = (float*) malloc(size);

    // Initialize matrices A and B
    init(h_a, N * N);
    init(h_b, N * N);

    // Allocate memory on device
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define block and grid size
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Launch the kernel
    mul<<<grid, block>>>(d_a, d_b, d_c, N);

    // Copy result from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print a few elements of the result matrix C as a sanity check
    for(int i = 0; i < 10; i++){
        printf("%f ", h_c[i]);
    }
    printf("\n");

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
