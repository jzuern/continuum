//
// Created by jannik on 5/18/18.
//


#ifndef NUMERICAL_KERNELS_H
#define NUMERICAL_KERNELS_H

#include <stdio.h>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32


#define gpuErrchk(ans) {gpuAssert ((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code!=cudaSuccess)
    {
        printf("CUDA failure %s:%s: '%d'\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


using namespace std;

__host__ __device__ int iDivUp(int a, int b);

void try_diffuse(float * x,float * x_old, int height, int width, const float diff, const float dt);
void try_source(float* x,float* s, float* x_result, int height, int width, const float dt);

void pretty_printer(float * x, int width, int height);

// kernel for 2D diffusion equation
__global__ void diffuse_kernel(float *x,float *x0, int height, int width);
__global__ void add_source_kernel(float *x,float *s, int height, int width);




#endif //NUMERICAL_KERNELS_H
