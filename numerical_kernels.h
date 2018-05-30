//
// Created by jannik on 5/18/18.
//


#ifndef NUMERICAL_KERNELS_H
#define NUMERICAL_KERNELS_H

#include <stdio.h>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>

#define TX 32
#define TY 32

using namespace std;

void try_diffuse(float* dens,float* dens_prev,int height, int width);
void pprinter(float * x, int width, int height);


// kernel for 2D diffusion equation
__global__ void diffuse_GPU(float *x,float *x0, int height, int width);




#endif //NUMERICAL_KERNELS_H
