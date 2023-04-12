#include<math.h>
#include<stdio.h>
#include <iostream>
#include <chrono>
int __device__ __host__ getn(int);
void __global__ add(int *data, int n) {
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < n)
    data[id] = getn(id);
  return;
}
int __device__ __host__ getn(int n) {
    if (n==0) return 0;
    else return n + getn(n-1);
}

void check(const int *z, const int N) {
  bool has_error = false;
  for (int n = 0; n < N; ++n) {
    if (z[n] != getn(n)) {
      has_error = true;
      printf("get c[%d]=%d\n", n, z[n]);
    }
  }
  printf("%s\n", has_error ? "Has errors" : "No errors");
}

int main(void) {
  std::cerr << "compile: " << __DATE__ << " | " << __TIME__ << std::endl;
  const int N = 100;
  const int M = sizeof(int) * N;
  int* data = (int *) malloc(M);
  
  int *d_data;
  cudaMalloc((void **) &d_data, M);

  auto t1 = std::chrono::high_resolution_clock::now();
  cudaMemcpy(d_data, data, M, cudaMemcpyHostToDevice);

  const int block_size = 50;
  const int grid_size = N / block_size;
  add<<<grid_size, block_size>>>(d_data, N);
  cudaMemcpy(data, d_data, M, cudaMemcpyDeviceToHost);
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "device time=" << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl;
  check(data, N);
  
  // ====cpu =====//
  /*
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
  
  // printf("data[%d]=%d, find id=%d\n", id, x, n+1);
*/

  free(data);
  cudaFree(d_data);
  return 0;
}

