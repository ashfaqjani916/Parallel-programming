#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#define N 20000000

void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

void populate(float *a, float *b, int n){
	for(int i = 0; i < n; i++){
        a[i] = rand()%1000; b[i] = rand()%1000;
    }
}
    
int main(){
    float *a, *b, *out; 

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    populate(a,b,N);
    clock_t start=clock();
    // Main function
    vector_add(out, a, b, N);
    clock_t stop=clock();
    for(int i = 0; i < N; i++){
	    printf("%f ",out[i]);
    }
    double timetaken= (double) (stop-start)/CLOCKS_PER_SEC;
    printf("\nThe time taken for Loop by CPU is: %lf\n",timetaken);
    return 0;
}
