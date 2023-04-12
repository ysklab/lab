#include<stdio.h>
__global__ void hello_from_gpu()
{
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int yid = threadIdx.y;
  printf("blockDim(xyz)=%d,%d,%d\n", blockDim.x, blockDim.y, blockDim.z);
  printf("hello word from block %d and thread (%d,%d)\n",bid,tid,yid);
}
int main()
{
  const dim3 block_size(2,4);
  hello_from_gpu<<<1,block_size>>>();
  cudaDeviceSynchronize();
  printf("helloword\n");
  return 0;
}

