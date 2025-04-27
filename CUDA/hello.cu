#include <cuda.h>
#include <stdio.h>

#define N 10

__global__ void printHello(){
  printf("Hello CUDA");
}

int main(){

            int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;

        printHello<<<gridSize,blockSize>>>();

        return 0;
    
}
