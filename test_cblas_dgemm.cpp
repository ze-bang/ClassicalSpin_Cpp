#include <cblas.h>
#include <stdio.h>
#include <array>
#include "simple_linear_alg.h"
int main()
{
  int i=0;
  std::array<double,2> A = {1.0,2.0};
  std::array<double,2> B = {1.0,2.0};
  std::array<double,2>  C = A+B*2;
  // printf("%lf\n", C);
  for(i=0; i<2; i++)
    printf("%lf ", C[i]);
  printf("\n");
  return 0;
}