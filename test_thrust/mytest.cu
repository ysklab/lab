#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <chrono>

struct Find
{
    __device__ __host__
    Find(thrust::device_vector<int>* a): arr(a) {} // seems wrong!!!
     __device__ __host__
        int operator()(const int& x) const {
            
            for (auto it = arr->begin(); it != arr->end(); ++it) {
                if (*it == x) { return it-arr->begin(); }
            }
            return arr->size() + 1;
        }
    thrust::device_vector<int>* arr;
};

int main(void)
{

    thrust::device_vector<int> a(10000), b(10000);
    thrust::sequence(a.begin(), a.end());

    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(0, 9999);
    thrust::device_vector<int> pick(999), out(999);
    for (int i =0; i < pick.size(); ++i) {
        pick[i] = dist(rng);
    }


    thrust::transform(pick.begin(), pick.end(), out.begin(), Find(&a));
    

    return 0;
}