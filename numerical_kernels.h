
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

void pretty_printer(float * x, int width, int height);

inline int get_idx(int i,int j, int NX);



// CFD ROUTINES

void try_diffuse(float * x,float * x_old, int height, int width, const float diff, const float dt, const int maxiter);
void try_source(float* x,float* s, int height, int width, const float dt);
void try_advect(float * d, float *  d0, float * u, float * v, const int height, const int width, const float dt, bool * occ);

void try_project_1(float * div, float *  u, float * v, float * p, const int height, const int width, const float h);
void try_project_2(float * p, float *  div, const int height, const int width, const int maxiter, bool * occ, float * dens, float * u);
void try_project_3(float * u, float *  v, float * p, const int height, const int width, const float h);

void try_set_bnd(int b, float * x, const int width, const int height, bool * occ, float * dens, float * u);


__global__ void diffuse_kernel(float *x,float *x0, int height, int width);
__global__ void add_source_kernel(float *x,float *s, int height, int width);
__global__ void advect_kernel(float * d_d,float * d_d0,float * d_u,float * d_v, int NX, int NY, float dt);

__global__ void project_kernel_1(float * d_div,float * d_u,float * d_v,float * d_p, int NX, int NY, float h);
__global__ void project_kernel_2(float * d_div,float * d_p, const int NX, const int NY);
__global__ void project_kernel_3(float * d_d,float * d_d0,float * d_u,float * d_v, int NX, int NY, float h);

__global__ void set_bnd_kernel(float * d_x, int NX, int NY, int b);



// NEW

void call_set_bnd_kernel(int b, float * x, const int width, const int height, dim3 dimGrid, dim3 dimBlock, bool * occ, float * dens, float * u);

void call_add_source_kernel(float * d_x, float * d_x0, const int NX, const int NY, const float dt, dim3 dimBlock, dim3 dimGrid);

void call_diffuse_kernel(float *x,float *s, int height, int width , float a, dim3 dimBlock, dim3 dimGrid);

void call_advect_kernel(float * d_d,float * d_d0,float * d_u,float * d_v, int NX, int NY, float dt, bool * occ, dim3 dimBlock, dim3 dimGrid);

void call_project_kernel_1(float * d_div,float * d_u,float * d_v,float * d_p, int NX, int NY, float h, dim3 dimBlock, dim3 dimGrid);

void call_project_kernel_2(float * d_div,float * d_p, const int NX, const int NY, dim3 dimBlock, dim3 dimGrid);

void call_project_kernel_3(float * d_d,float * d_d0,float * d_u,float * d_v, int NX, int NY, float h, dim3 dimBlock, dim3 dimGrid);

#endif //NUMERICAL_KERNELS_H
