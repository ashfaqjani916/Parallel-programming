
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

void main(){
    int a[1000], b[1000], c[1000], i, j;
    double start_time, end_time;

    for (i = 0; i < 1000; i++) {
        a[i] = rand();
        b[i] = rand();
    }

    omp_set_num_threads(100);

    start_time = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp for
        for (j = 0; j < 1000; j++) {
            c[j] = a[j] + b[j]; 
        }
    }

    end_time = omp_get_wtime();

    printf("Time taken: %f seconds\n", end_time - start_time);
    printf("Timer resolution: %f seconds\n", omp_get_wtick());
}
