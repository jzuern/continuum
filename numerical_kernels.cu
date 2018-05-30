
#include <cuda_runtime.h>
#include <cuda.h>
#include "numerical_kernels.h"




__global__ void diffuse_GPU(float *dens, float *dens_prev, int height, int width)
{
    // diffusion step is obtained by Gauss-Seidel relaxation equation system solver
    // used for density, u-component and v-component of velocity field separately
    float diff = 0.01;
    const float dt = 0.0001; // incremental time step length

    float a = dt*diff*height*width;

//    printf("in diffuse_GPU\n");

    // indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i> 1 and i < (width+1) and j > 1 and j < (height+2))
    {
        printf("dens_prev [IX(%i, %i)] = %f\n", i,j,dens_prev[IX(i, j)]);

        for (int k=0 ; k < 10 ; k++ ) {

            dens[IX(i, j)] = (dens_prev[IX(i, j)] +
                              a * (dens[IX(i - 1, j)] + dens[IX(i + 1, j)] + dens[IX(i, j - 1)] + dens[IX(i, j + 1)])) /
                             (1 + 4 * a);

        }
    }

}


void try_diffuse(float* dens,float* dens_prev,int height, int width)
{
    // Copy data to device


    int size = (height+2) * (width+2);


    float *dens_d , *dens_prev_d; // device array

    //create device array cudaMalloc ( (void **)&array_name, sizeofmatrixinbytes) ;
    cudaMalloc((void **) &dens_d , size*sizeof (float) ) ;
    cudaMalloc((void **) &dens_prev_d , size*sizeof (float) ) ;

    //copy host array to device array; cudaMemcpy ( dest , source , WIDTH , direction )
    cudaMemcpy ( dens_d , dens , size*sizeof (float) , cudaMemcpyHostToDevice ) ;
    cudaMemcpy ( dens_prev_d , dens_prev , size*sizeof (float) , cudaMemcpyHostToDevice ) ;


    // Blocks and grid galore

    dim3 threadsPerBlock(100,10);
    dim3 numBlocks(size / threadsPerBlock.x, size / threadsPerBlock.y);

//    dim3 dimThreadBlock (TX, TY);
//    dim3 dimBlockGrid (size/TX,1);

    printf(" dens before heat diffusion: \n");
    pprinter(dens, height, width);

    diffuse_GPU <<< numBlocks, threadsPerBlock >>> (dens_d, dens_prev_d, height, width) ;

    cudaThreadSynchronize();

    // all gpu function blocked till kernel is working
    //copy back result_array_d to result_array_h

    cudaMemcpy(dens , dens_d , size*sizeof(float) , cudaMemcpyDeviceToHost) ;

    printf(" dens after heat diffusion: \n");
    pprinter(dens,height, width);

}


void pprinter(float * x, int width, int height)
{
    for (int i=0 ; i<=width+1 ; i++ )
    {
        for (int j=0 ; j<=height+1 ; j++ )
        {
            printf("(%i,%i): %f ", i,j,x[IX(i,j)] );
        }
        printf("\n");
    }
    printf("\n\n\n");
}