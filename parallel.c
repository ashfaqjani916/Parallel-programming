#include <stdio.h>
#include <omp.h>

void main()
{
  #pragma omp parallel
  {
    int threadID = omp_get_thread_num();
    printf("this is thread:%d\n",threadID);
  }
}
