
#include <cuda_runtime.h>
#include <cuda.h>
#include "numerical_kernels.h"

__host__ __device__ int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }


__global__ void diffuse_kernel(float * x, const float * x_old, const int NX, const int NY, const float a)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x ;
    const int j = blockIdx.y * blockDim.y + threadIdx.y ;

    //                                                     N
    int P = i + j*NX;           // node (i,j)              |
    int N = i + (j+1)*NX;       // node (i,j+1)            |
    int S = i + (j-1)*NX;       // node (i,j-1)     W ---- P ---- E
    int E = (i+1) + j*NX;       // node (i+1,j)            |
    int W = (i-1) + j*NX;       // node (i-1,j)            |
    //                                                     S

    // --- Only update "interior" (not boundary) node points
    if (i>0 && i<NX-1 && j>0 && j<NY-1)
    {
        x[P] = (x_old[P] + a * (x[W] + x[E] + x[S] + x[N])) / (1 + 4 * a);
    }
}



__global__ void add_source_kernel(float * x, const float * s, const int NX, const int NY, const float dt)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x ;
    const int j = blockIdx.y * blockDim.y + threadIdx.y ;

    int P = i + j*NX;

    // --- Only update "interior" (not boundary) node points
    if (i>0 && i<NX-1 && j>0 && j<NY-1)
    {
        x[P] += dt*s[P];
    }
}


void try_diffuse(float* x,float* x_prev, int height, int width, const float diff, const float dt)
{

    const int NX = width+2;		// --- Number of discretization points along the x axis
    const int NY = height+2;    // --- Number of discretization points along the y axis
    const int MAX_ITER = 10;	// --- Number of Jacobi iterations
    const float a = dt*diff*height*width;


    // allocate cuda memory
    float *x_d;
    gpuErrchk(cudaMalloc((void**)&x_d, NX * NY * sizeof(float)));
    float *x_prev_d;
    gpuErrchk(cudaMalloc((void**)&x_prev_d, NX * NY * sizeof(float)));

    // copy host memory to device memory
    gpuErrchk(cudaMemcpy(x_d,			x,	 NX * NY * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(x_prev_d,		x_d,	 NX * NY * sizeof(float), cudaMemcpyDeviceToDevice));


    // --- Grid size
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 dimGrid (iDivUp(NX, BLOCK_SIZE_X), iDivUp(NY, BLOCK_SIZE_Y));

    for (int k=0; k<MAX_ITER; k++)
    {
        diffuse_kernel <<< dimGrid, dimBlock >>> (x_d, x_prev_d, NX, NY, a);
    }

    // --- Copy results from device to host
    gpuErrchk(cudaMemcpy(x_prev, x_d, NX * NY * sizeof(float), cudaMemcpyDeviceToHost));

    // free device memory
    gpuErrchk(cudaFree(x_d));
    gpuErrchk(cudaFree(x_prev_d));
}


void try_source(float* x,float* s, float* h_T_GPU_result, int height, int width, const float dt)
{

    const int NX = width+2;			// --- Number of discretization points along the x axis
    const int NY = height+2;			// --- Number of discretization points along the y axis

    // allocate cuda memory
    float *d_x;			gpuErrchk(cudaMalloc((void**)&d_x,			NX * NY * sizeof(float)));
    float *d_s;			gpuErrchk(cudaMalloc((void**)&d_s,			NX * NY * sizeof(float)));

    // copy host memory to device memory
    gpuErrchk(cudaMemcpy(d_x, x,	 NX * NY * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_s, s,	 NX * NY * sizeof(float), cudaMemcpyHostToDevice));

    // --- Grid size
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 dimGrid (iDivUp(NX, BLOCK_SIZE_X), iDivUp(NY, BLOCK_SIZE_Y));

    add_source_kernel<<<dimGrid, dimBlock>>>(d_x,d_s, NX, NY,dt);   // --- Update d_T_old     starting from data stored in d_T

    // --- Copy results from device to host
    gpuErrchk(cudaMemcpy(h_T_GPU_result,	 d_x,	  NX * NY * sizeof(float), cudaMemcpyDeviceToHost));

    // free device memory
    gpuErrchk(cudaFree(d_x));
    gpuErrchk(cudaFree(d_s));

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