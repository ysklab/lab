#include<math.h>
#include<stdio.h>
#include <iostream>
#include <chrono>
void __global__ add(const int *data, const int *arr, int n, int *r) {
  const int id = blockDim.x * blockIdx.x + threadIdx.x;
  int x= data[id];
  for (int i = 0; i< n; ++i) {
    if (arr[i] == x) {
      r[id] = i;
      // printf("data[%d]=%d, find id=%d\n", id, x, i);
      return;
    }
  }
  // printf("data[%d]=%d, find id=%d\n", id, x, n+1);
  r[id] = n+1;
  return;

}
void check(const int *z, const int N) {
  bool has_error = false;
  for (int n = 0; n < N; ++n) {
    if (z[n] != n) {
      has_error = true;
      printf("get c[%d]=%d\n", n, z[n]);
    }
  }
  printf("%s\n", has_error ? "Has errors" : "No errors");
}

int main(void) {
  std::cerr << "compile: " << __DATE__ << " | " << __TIME__ << std::endl;
  const int N = 100000;
  const int M = sizeof(int) * N;
  int* data = (int *) malloc(M);
  int* arr = (int *) malloc(M);
  int* r = (int *) malloc(M);
  for (int n = 0; n < N; ++n) {
    data[n] = n;
    arr[n] = n;
    r[n] = 0;
  }
  int *d_data, *d_arr, *d_r;
  cudaMalloc((void **) &d_data, M);
  cudaMalloc((void **) &d_arr, M);
  cudaMalloc((void **) &d_r, M);

  auto t1 = std::chrono::high_resolution_clock::now();
  cudaMemcpy(d_data, data, M, cudaMemcpyHostToDevice);
  cudaMemcpy(d_arr, arr, M, cudaMemcpyHostToDevice);
  const int block_size = 50;
  const int grid_size = N / block_size;
  add<<<grid_size, block_size>>>(d_data, d_arr, N, d_r);
  cudaMemcpy(r, d_r, M, cudaMemcpyDeviceToHost);
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "device time=" << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl;

  check(r, N);
  // ====cpu =====//
  auto tt1 = std::chrono::high_resolution_clock::now();
  for (int n = 0; n <N; ++n) {
    int x = data[n];
    for (int i = 0; i< N; ++i) {
      if (arr[i] == x) {
        r[n] = i;
        // printf("data[%d]=%d, find id=%d\n", id, x, i);
        break;
      }
    }
  }
  auto tt2 = std::chrono::high_resolution_clock::now();
  std::cout << "host time=" << std::chrono::duration_cast<std::chrono::milliseconds>(tt2-tt1).count() << std::endl;
  check(r, N);
  // printf("data[%d]=%d, find id=%d\n", id, x, n+1);


  free(data);
  free(arr);
  free(r);
  cudaFree(d_data);
  cudaFree(d_arr);
  cudaFree(d_r);
  return 0;
}

