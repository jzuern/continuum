#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>
#include <gtkmm.h>

#include "simulation.h"
#include "numerical_kernels.h"


// CUDA STUFF
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE_X 1
#define BLOCK_SIZE_Y 1

using namespace std;


__global__ void DiffX_GPU(float* d_U, float* d_Ux, int N, int alpha, float* d_stencils, int rank)

{
    printf("hello from thread %i\n", threadIdx.x);


    // indices
     const int b_i =   threadIdx.x;
     const int b_j = blockIdx.y*blockDim.y + threadIdx.y;
     const int n = b_i * N + b_j;

    int row;
    for (row=0; row<N; ++row)
    {
        float value=0.0;
        // Compute dot-product between FDM stencil weights and input vector U
        int diff = 0; // diff is used for automatically taking one-sided difference near boundaries
        if (row<alpha)
            diff = alpha - row;
        else if (row>N-1-alpha)  // row  >   Nx-3 Nx-2 Nx-1
            diff = N-1-alpha-row;
        int tmp = (alpha-diff)*rank+alpha;
        int tmp2 = row + diff;
        int i;
        for (i = -alpha; i<alpha+1; ++i)
            value += (d_U[tmp2+i]*d_stencils[tmp+i])  ;
        // Store computed approximation
        d_Ux[row] =   value;
    }
}


int main(int argc, char* argv[])
{


//    int Nx = 100;
//    int rank = 10;
//    int alpha = 1;
//
//    // Allocate space on device
//    float *d_U, *d_Ux, *d_stencils;
//
//    float *U = new float[Nx];
//    float *Ux = new float[Nx];
//    float *stencils = new float[rank];
//
//    cudaMalloc ((void**) &d_U, Nx*sizeof(float)); // TODO: Error checking
//    cudaMalloc ((void**) &d_Ux, Nx*sizeof(float)); // TODO: Error checking
//    cudaMalloc ((void**) &d_stencils, rank*sizeof(float)); // TODO: Error checking
//
//    // Copy data to device
//    cudaMemcpy (d_U, U, Nx*sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy (d_stencils, stencils, rank*sizeof(float), cudaMemcpyHostToDevice);
//
//    // Blocks and grid galore
//    dim3 dimThreadBlock (BLOCK_SIZE_X, BLOCK_SIZE_Y);
//    dim3 dimBlockGrid (Nx/BLOCK_SIZE_X,1);
//
//    DiffX_GPU <<< dimBlockGrid, dimThreadBlock >>> (d_U, d_Ux,  Nx, alpha, d_stencils, rank) ;
//
//    cudaThreadSynchronize();
//
//    // Copy result to host
//
//    cudaMemcpy (Ux, d_Ux, Nx*sizeof(float), cudaMemcpyDeviceToHost);




    // open GUI window
    Glib::RefPtr<Gtk::Application> app = Gtk::Application::create(argc, argv, "continuum.de");
    
    // initiate simulation instance
    Simulation sim;
    
    
    //Run the app with a simulation
    return app->run(sim);
}


