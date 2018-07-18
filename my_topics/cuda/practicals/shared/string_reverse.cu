#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "util.hpp"

// TODO : implement a kernel that reverses a string of length n in place
// Use shared memory, even if you don't really need it here for practice
__global__
void reverse_string(char* str, int n){
    __shared__ char buffer[1024];
    int ind = threadIdx.x + blockDim.x*blockIdx.x;
    if (ind<n){
        buffer[ind] = str[n-ind-1];
        }
    __syncthreads();
    if (ind<n){
        str[ind] = buffer[ind];
        }
    }

// kernel that reverses a string of length n in place
// no shared memory, no synch
// note that nthreads is defined differently
__global__
void reverse_string_noshared(char* str, int n){
    int ind = threadIdx.x + blockDim.x*blockIdx.x;
    char temp = str[ind];
    str[ind] = str[n-ind-1];
    str[n-ind-1] = temp;
    }


int main(int argc, char** argv) {
    // check that the user has passed a string to reverse
    if(argc<2) {
        std::cout << "useage : ./string_reverse \"string to reverse\"\n" << std::endl;
        exit(0);
    }

    // determine the length of the string, and copy in to buffer
    auto n = strlen(argv[1]);
    auto string = malloc_managed<char>(n+1);
    std::copy(argv[1], argv[1]+n, string);
    string[n] = 0; // add null terminator

    std::cout << "string to reverse:\n" << string << "\n";

    // TODO : call the string reverse function
    /* int nthreads=128; */
    /* int nblocks =(n/128+1); */
    /* reverse_string<<<nthreads,nblocks>>>(string, n); */

    int nthreads=128;
    int nblocks=(n/128+1)/2+1;
    std::cout << nthreads << " " << nblocks << "\n";
    reverse_string_noshared<<<1,n/2>>>(string,n);

    // print reversed string
    cudaDeviceSynchronize();
    std::cout << "reversed string:\n" << string << "\n";

    // free memory
    cudaFree(string);

    return 0;
}

