#include <cblas.h>
#include <stdio.h>
#include <array>
int main()
{
  int i=0;
  std::array<double,6> A = {1.0,2.0,1.0,-3.0,4.0,-1.0};         
  std::array<double,6> B = {1.0,2.0,1.0,-3.0,4.0,-1.0};  
  std::array<double,9> C = {.5,.5,.5,.5,.5,.5,.5,.5,.5}; 
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,3,3,2,1, A.data(), 3, B.data(), 3, 2,C.data(),3);

  for(i=0; i<9; i++)
    printf("%lf ", C[i]);
  printf("\n");
  return 0;
}