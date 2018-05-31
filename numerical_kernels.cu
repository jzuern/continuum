
#include <cuda_runtime.h>
#include <cuda.h>
#include "numerical_kernels.h"

__host__ __device__ int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }


__global__ void diffuse_GPU(float *dens, float *dens_prev, int height, int width)
{
    // diffusion step is obtained by Gauss-Seidel relaxation equation system solver
    // used for density, u-component and v-component of velocity field separately
    float diff = 0.01;
    const float dt = 0.0001; // incremental time step length

    float a = dt*diff*height*width;

    // indices
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    // directions

    int P = i + j*(width+2);
    int N = i + (j+1)*(width+2);
    int S = i + (j-1)*(width+2);
    int E = (i+1) + j*(width+2);
    int W = (i-1) + j*(width+2);

    if (i > 0 and i < (width+2) and j > 0 and j < (height+2))
        dens [P] = dens_prev[P] + a * (dens[W] + dens[S] + dens[N] + dens[E]) / (1 + 4 * a);


//    // only update interior
//    if (i > 0 and i < (width+2) and j > 0 and j < (height+2))
//    {
//        printf("dens_prev1 [IX(%i, %i)] = %f\n", i,j,dens_prev[IX(i, j)]);
//        printf("dens1 [IX(%i, %i)] = %f\n", i,j,dens[IX(i, j)]);
//
////        for (int k=0 ; k < 10 ; k++ ) {
//
//            dens[IX(i, j)] = (dens_prev[IX(i, j)] +
//                              a * (dens[IX(i - 1, j)] + dens[IX(i + 1, j)] + dens[IX(i, j - 1)] + dens[IX(i, j + 1)])) /
//                             (1 + 4 * a);
//
////        }
//        printf("dens_prev2 [IX(%i, %i)] = %f\n", i,j,dens_prev[IX(i, j)]);
//        printf("dens2 [IX(%i, %i)] = %f\n", i,j,dens[IX(i, j)]);
//
//    }
}

__global__ void Jacobi_Iterator_GPU(const float * __restrict__ T_old, float * __restrict__ T_new, const int NX, const int NY)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x ;
    const int j = blockIdx.y * blockDim.y + threadIdx.y ;

    //                         N
    int P = i + j*NX;           // node (i,j)              |
    int N = i + (j+1)*NX;       // node (i,j+1)            |
    int S = i + (j-1)*NX;       // node (i,j-1)     W ---- P ---- E
    int E = (i+1) + j*NX;       // node (i+1,j)            |
    int W = (i-1) + j*NX;       // node (i-1,j)            |
    //                         S

    // --- Only update "interior" (not boundary) node points
    if (i>0 && i<NX-1 && j>0 && j<NY-1) T_new[P] = 0.25 * (T_old[E] + T_old[W] + T_old[N] + T_old[S]);
}


void try_diffuse(float* dens,float* dens_prev, float* h_T_GPU_result, int height, int width)
{

    const int NX = width+2;			// --- Number of discretization points along the x axis
    const int NY = height+2;			// --- Number of discretization points along the y axis
    const int MAX_ITER = 10;	// --- Number of Jacobi iterations


    // allocate cuda memory
    float *d_T;			gpuErrchk(cudaMalloc((void**)&d_T,			NX * NY * sizeof(float)));
    float *d_T_old;		gpuErrchk(cudaMalloc((void**)&d_T_old,		NX * NY * sizeof(float)));

    // copy host memory to device memory
    gpuErrchk(cudaMemcpy(d_T,			dens,	 NX * NY * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_T_old,		d_T,	 NX * NY * sizeof(float), cudaMemcpyDeviceToDevice));


    // --- Grid size
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 dimGrid (iDivUp(NX, BLOCK_SIZE_X), iDivUp(NY, BLOCK_SIZE_Y));


    for (int k=0; k<MAX_ITER; k=k+2) {
        Jacobi_Iterator_GPU<<<dimGrid, dimBlock>>>(d_T,     d_T_old, NX, NY);   // --- Update d_T_old     starting from data stored in d_T
        Jacobi_Iterator_GPU<<<dimGrid, dimBlock>>>(d_T_old, d_T    , NX, NY);   // --- Update d_T         starting from data stored in d_T_old
    }


    // --- Copy results from device to host
    gpuErrchk(cudaMemcpy(h_T_GPU_result,	 d_T,	  NX * NY * sizeof(float), cudaMemcpyDeviceToHost));

    // free device memory
    gpuErrchk(cudaFree(d_T));
    gpuErrchk(cudaFree(d_T_old));

}


void pretty_printer(float * x, int width, int height)
{
    for (int i=0 ; i < width+2; i++)
    {
        for (int j=0 ; j < height+2; j++)
        {
            printf("(%i,%i): %f ", i,j,x[IX(i,j)]);
        }
        printf("\n");
    }
    printf("\n\n\n");
}