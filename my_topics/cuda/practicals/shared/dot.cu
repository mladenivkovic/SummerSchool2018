#include <iostream>

#include <cuda.h>

#include "util.hpp"

// host implementation of dot product
double dot_host(const double *x, const double* y, int n) {
    double sum = 0;
    for(auto i=0; i<n; ++i) {
        sum += x[i]*y[i];
    }
    return sum;
}

// TODO implement dot product kernel
template <int THREADS>
__global__
void dot_gpu_kernel(const double *x, const double* y, double *result, int n) {
    // computes the multiplication of two vectors x and y
    int ind = threadIdx.x + blockIdx.x*blockDim.x;
    if (ind<n) {
        result[ind] = x[ind]+y[ind];
    }
}

__global__
void reduce_array(double *x, double *result, int n){
    // assuming I run on n/2 threads
    // sums up 2 elements of array x
    int ind_res = (threadIdx.x + blockIdx.x*blockDim.x);
    int ind_x = 2*ind_res;
    if (ind_x < n-1){
        result[ind_res] = x[ind_x]+x[ind_x+1];
         }
    else {
        result[ind_res] = x[ind_x];
    }
}



double dot_gpu(const double *x, const double* y, int n) {
    static double* result = malloc_managed<double>(1);
    double* result_arr = malloc_managed<double>(n);
    double* result_arr2 = malloc_managed<double>(n/2+1);
    // TODO call dot product kernel
    dot_gpu_kernel<1024><<<1,n>>>(x, y, result_arr, n);
    while (n>1){
        n = n/2+1;
        reduce_array<<<1,n>>>(result_arr, result_arr2, n);
    }
    result = &result_arr[0];



    cudaDeviceSynchronize();
    return *result;
}


//==================================================
template <int THREADS>
__global__
void dot_gpu_kernel_solution(const double *x, const double* y, double *result, int n) {
    __shared__ double buf[1024];

    int i = threadIdx.x;

    buf[i] = 0;
    if (i<n){
        buf[i] = x[i]*y[i];
    }

    int m = THREADS/2;

    while(m) {
        __syncthreads();
        if (i<m) {
            buf[i] += buf[i+m];
        }
        m = m/2;
    }

    if (i==0) {
        *result = buf[0];
    }
}


double dot_gpu_solution(const double *x, const double *y, int n){
    static double* result = malloc_managed<double>(1);
    dot_gpu_kernel_solution<1024><<<1,1024>>>(x,y,result,n);

    cudaDeviceSynchronize();
    return *result;


}





// solution for arbitrary number of threads and blocks
double dot_gpu_solution_arbitrary(const double *x, const double *y, int n){
    static double* result = malloc_managed<double>(1);
    *results = 0;
    constexpr int block_dim = 1024;
    dot_gpu_kernel_solution<blockdim><<<(n+block_dim-1)/block_dim, block_dim>>>(x,y,result,n);

    cudaDeviceSynchronize();
    return *result;
}


// solution for arbitrary number of threads and blocks
template <int THREADS>
__global__
void dot_gpu_kernel_solution_arbitrary(const double *x, const double* y, double *result, int n) {
    // todo: not complete solution
    __shared__ double buf[1024];

    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    int i = threadIdx.x;

    buf[i] = 0;
    if (gid<n){
        buf[i] = x[gidi]*y[gid];
    }

    int m = THREADS/2;

    while(m) {
        __syncthreads();
        if (i<m) {
            buf[i] += buf[i+m];
        }
        m = m/2;
    }

    if (i==0) {
        atomicAdd(result, buf[0]);
    }
}

//==================================================


int main(int argc, char** argv) {
    size_t pow = read_arg(argc, argv, 1, 4);
    size_t n = (1 << pow);

    auto size_in_bytes = n * sizeof(double);

    std::cout << "dot product CUDA of length n = " << n
              << " : " << size_in_bytes*1e-9 << "MB\n";

    auto x_h = malloc_host<double>(n, 2.);
    auto y_h = malloc_host<double>(n);
    for(auto i=0; i<n; ++i) {
        y_h[i] = rand()%10;
    }

    auto x_d = malloc_device<double>(n);
    auto y_d = malloc_device<double>(n);

    // copy initial conditions to device
    copy_to_device<double>(x_h, x_d, n);
    copy_to_device<double>(y_h, y_d, n);

    auto result   = dot_gpu_solution(x_d, y_d, n);
    auto expected = dot_host(x_h, y_h, n);
    printf("expected %f got %f\n", (float)expected, (float)result);

    return 0;
}

