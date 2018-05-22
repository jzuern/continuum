//
// Created by jannik on 5/18/18.
//


#ifndef CUDA_SQUARE_H
#define CUDA_SQUARE_H

#include <stdio.h>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda.h>



using namespace std;

__global__ void heat_kernel( float *data_array_new_device , float *data_array_old_device , const int width );
void launch_kernel();
void cuda_init();


// testing

__global__ void MatrixMul( float *Md , float *Nd , float *Pd , const int WIDTH );
void try_cuda();

void test();


#endif //CUDA_SQUARE_H
