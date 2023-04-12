#include<math.h>
#include<stdio.h>
#include <chrono>
#include <iostream>
const double EPSILON = 1.0e-5;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void __global__ add(const double *x, const double *y, double *z);
void check(const double *z, const int N);
int main(void) {
  const int N = 100000000;
  const int M = sizeof(double) * N;
  double *h_x = (double *) malloc(M);
  double *h_y = (double *) malloc(M);
  double *h_z = (double *) malloc(M);
  for (int n = 0; n < N; ++n) {
    h_x[n] = a;
    h_y[n] = b;
  }
  double *d_x, *d_y, *d_z;
  cudaMalloc((void **) &d_x, M);
  cudaMalloc((void **) &d_y, M);
  cudaMalloc((void **) &d_z, M);
  auto t1 = std::chrono::high_resolution_clock::now();
  cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);
  const int block_size = 128;
  const int grid_size = N / block_size;
  add<<<grid_size, block_size>>>(d_x, d_y, d_z);
  cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "device time=" << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl;
  check(h_z, N);
  auto tt1 = std::chrono::high_resolution_clock::now();
  for (int n = 0; n<N; ++n) {
    h_z[n] = h_x[n] + h_y[n];
  }
  auto tt2 = std::chrono::high_resolution_clock::now();
  std::cout << "host time=" << std::chrono::duration_cast<std::chrono::milliseconds>(tt2-tt1).count() << std::endl;
  free(h_x);
  free(h_y);
  free(h_z);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  return 0;
}
void __global__ add(const double *x, const double *y, double *z) {
  const int n = blockDim.x * blockIdx.x + threadIdx.x;
  z[n] = x[n] + y[n];
  if (fabs(z[n] - c)> EPSILON ) {
    printf("%f+%f=%f\n", x[n], y[n], z[n]);
  }
}
void check(const double *z, const int N) {
  bool has_error = false;
  for (int n = 0; n < N; ++n) {
    if (fabs(z[n] - c) > EPSILON) {
      has_error = true;
      printf("get c[%d]=%f\n", n, z[n]);
    }
  }
  printf("%s\n", has_error ? "Has errors" : "No errors");
}
