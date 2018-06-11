
#include <cuda_runtime.h>
#include <cuda.h>
#include "numerical_kernels.h"


#define get_idx(i,j,NX) ((i)+(NX+2)*(j))



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




__global__ void advect_kernel(float * d,float * d0,float * u, float * v, int NX, int NY, float dt, bool * occ)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x ;
    const int j = blockIdx.y * blockDim.y + threadIdx.y ;

    int idx = i + j*NX;

    int i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;
    dt0 = dt*max(NX,NY);

    // --- Only update "interior" (not boundary) node points
    if (i>0 && i<NX-1 && j>0 && j<NY-1)
    {

        x = i-dt0*u[idx];
        y = j-dt0*v[idx];

        if (x<0.5) x=0.5;
        if (x>NX+0.5) x = NX + 0.5; i0=(int)x; i1=i0+ 1;
        if (y<0.5) y=0.5;
        if (y>NY+0.5) y = NY + 0.5; j0=(int)y; j1=j0+1;

        s1 = x-i0;
        s0 = 1-s1;
        t1 = y-j0;
        t0 = 1-t1;

        int idx00 = i0 + j0*NX;
        int idx11 = i1 + j1*NX;
        int idx01 = i0 + j1*NX;
        int idx10 = i1 + j0*NX;

        if (occ[idx]==false)
            d[idx] = s0*(t0*d0[idx00] + t1*d0[idx01]) + s1*(t0*d0[idx10] + t1*d0[idx11]);
        else
            d[idx] = 0.0;
    }
}



__global__ void project_kernel_1(float * div ,float * u,float * v, float * p, int NX, int NY, const float h)
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
        div[P] = -0.5*h*(u[E]-u[W]+v[N]-v[S]);
        p[P] = 0;
    }
}


__global__ void project_kernel_2(float * div,float * p, const int NX, const int NY)
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
        p[P] = (div[P]+p[W]+p[E]+p[S]+p[N])/4;
    }
}

__global__ void project_kernel_3(float * u,float * v,float * p, int NX, int NY, const float h)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x ;
    const int j = blockIdx.y * blockDim.y + threadIdx.y ;


    //                                                     N
    int P = i + j*NX;           // node (i,j)              |
    int N = i + (j+1)*NX;       // node (i,j+1)            |
    int S = i + (j-1)*NX;       // node (i,j-1)     W ---- P ---- E
    int E = (i+1) + j*NX;       // node (i+1,j)            |
    int W = (i-1) + j*NX;       // node (i-1,j)            |
    //


    // --- Only update "interior" (not boundary) node points
    if (i>0 && i<NX-1 && j>0 && j<NY-1)
    {
        u[P] -= 0.5*(p[E]-p[W])/h;
        v[P] -= 0.5*(p[N]-p[S])/h;
    }
}



void try_diffuse(float* x,float* x_prev, int height, int width, const float diff, const float dt, const int maxiter)
{

    const int NX = width+2;		// --- Number of discretization points along the x axis
    const int NY = height+2;    // --- Number of discretization points along the y axis
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

    for (int k=0; k<maxiter; k++)
    {
        diffuse_kernel <<< dimGrid, dimBlock >>> (x_d, x_prev_d, NX, NY, a);
    }

    // --- Copy results from device to host
    gpuErrchk(cudaMemcpy(x_prev, x_d, NX * NY * sizeof(float), cudaMemcpyDeviceToHost));

    // free device memory
    gpuErrchk(cudaFree(x_d));
    gpuErrchk(cudaFree(x_prev_d));
}


void try_source(float* x,float* s, int height, int width, const float dt)
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
    gpuErrchk(cudaMemcpy(x,	 d_x,	  NX * NY * sizeof(float), cudaMemcpyDeviceToHost));

    // free device memory
    gpuErrchk(cudaFree(d_x));
    gpuErrchk(cudaFree(d_s));

}


void try_advect(float * d, float *  d0, float * u, float * v, const int height, const int width, const float dt, bool * occ)
{
    const int NX = width+2;			// --- Number of discretization points along the x axis
    const int NY = height+2;			// --- Number of discretization points along the y axis

    // allocate cuda memory
    float *d_d;			gpuErrchk(cudaMalloc((void**)&d_d,			NX * NY * sizeof(float)));
    float *d_d0;		gpuErrchk(cudaMalloc((void**)&d_d0,			NX * NY * sizeof(float)));
    float *d_u;			gpuErrchk(cudaMalloc((void**)&d_u,			NX * NY * sizeof(float)));
    float *d_v;			gpuErrchk(cudaMalloc((void**)&d_v,			NX * NY * sizeof(float)));
    bool *d_occ;		gpuErrchk(cudaMalloc((void**)&d_occ,			NX * NY * sizeof(float)));

    // copy host memory to device memory
    gpuErrchk(cudaMemcpy(d_d0, d0,	 NX * NY * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, u,	 NX * NY * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v, v,	 NX * NY * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_occ,occ,	 NX * NY * sizeof(bool), cudaMemcpyHostToDevice));

    // --- Grid size
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 dimGrid (iDivUp(NX, BLOCK_SIZE_X), iDivUp(NY, BLOCK_SIZE_Y));

    advect_kernel<<<dimGrid, dimBlock>>>(d_d,d_d0,d_u,d_v, NX, NY,dt, d_occ);

    // --- Copy results from device to host
    gpuErrchk(cudaMemcpy(d,	 d_d,	  NX * NY * sizeof(float), cudaMemcpyDeviceToHost));

    // free device memory
    gpuErrchk(cudaFree(d_d));
    gpuErrchk(cudaFree(d_d0));
    gpuErrchk(cudaFree(d_u));
    gpuErrchk(cudaFree(d_v));
    gpuErrchk(cudaFree(d_occ));

}


//void try_project_1(float * div, float * u,float * v,float * p, const int height, const int width, const float h)
//{
//    const int NX = width+2;			// --- Number of discretization points along the x axis
//    const int NY = height+2;			// --- Number of discretization points along the y axis
//
//    // allocate cuda memory
//    float *d_div;	    gpuErrchk(cudaMalloc((void**)&d_div,			NX * NY * sizeof(float)));
//    float *d_u;		    gpuErrchk(cudaMalloc((void**)&d_u,			NX * NY * sizeof(float)));
//    float *d_v;			gpuErrchk(cudaMalloc((void**)&d_v,			NX * NY * sizeof(float)));
//    float *d_p;			gpuErrchk(cudaMalloc((void**)&d_p,			NX * NY * sizeof(float)));
//
//    // copy host memory to device memory
//    gpuErrchk(cudaMemcpy(d_u, u,	 NX * NY * sizeof(float), cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_v, v,	 NX * NY * sizeof(float), cudaMemcpyHostToDevice));
//
//
//    // Grid size
//    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
//    dim3 dimGrid (iDivUp(NX, BLOCK_SIZE_X), iDivUp(NY, BLOCK_SIZE_Y));
//
//    project_kernel_1<<<dimGrid, dimBlock>>>(d_div,d_u,d_v,d_p, NX, NY, h);
//
//
//    // --- Copy results from device to host
//    gpuErrchk(cudaMemcpy(div,d_div,	  NX * NY * sizeof(float), cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(p,	 d_p,	  NX * NY * sizeof(float), cudaMemcpyDeviceToHost));
//
//    // free device memory
//    gpuErrchk(cudaFree(d_div));
//    gpuErrchk(cudaFree(d_u));
//    gpuErrchk(cudaFree(d_v));
//    gpuErrchk(cudaFree(d_p));
//
//}


//
//void try_project_2(float * div, float * p, const int height, const int width, const int maxiter, bool * occ, float * dens, float * u)
//{
//    const int NX = width+2;			// --- Number of discretization points along the x axis
//    const int NY = height+2;			// --- Number of discretization points along the y axis
//
//    // allocate cuda memory
//    float *d_div;	    gpuErrchk(cudaMalloc((void**)&d_div,		NX * NY * sizeof(float)));
//    float *d_p;		    gpuErrchk(cudaMalloc((void**)&d_p,			NX * NY * sizeof(float)));
//
//    // copy host memory to device memory
//    gpuErrchk(cudaMemcpy(d_div, div,	 NX * NY * sizeof(float), cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_p, p,	 NX * NY * sizeof(float), cudaMemcpyHostToDevice));
//
//
//    // Grid size
//    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
//    dim3 dimGrid (iDivUp(NX, BLOCK_SIZE_X), iDivUp(NY, BLOCK_SIZE_Y));
//
//    for (int k=0; k<maxiter; k++)
//    {
//        project_kernel_2 << < dimGrid, dimBlock >> > (d_div, d_p, NX, NY);
//
//        // --- Copy results from device to host
//        gpuErrchk(cudaMemcpy(p,d_p,	  NX * NY * sizeof(float), cudaMemcpyDeviceToHost));
//        //set_bnd_cp(0, p, NX,NY, occ);
//        try_set_bnd(0,p,width,height,occ,dens,u);
//    }
//
//    // --- Copy results from device to host
//    gpuErrchk(cudaMemcpy(p,d_p,	  NX * NY * sizeof(float), cudaMemcpyDeviceToHost));
//
//    // free device memory
//    gpuErrchk(cudaFree(d_p));
//    gpuErrchk(cudaFree(d_div));
//}
//
//
//void try_project_3(float * u, float *  v, float * p, const int height, const int width, const float h)
//{
//    const int NX = width+2;			    // --- Number of discretization points along the x axis
//    const int NY = height+2;			// --- Number of discretization points along the y axis
//
//    // allocate cuda memory
//    float *d_u;	        gpuErrchk(cudaMalloc((void**)&d_u,		    NX * NY * sizeof(float)));
//    float *d_v;		    gpuErrchk(cudaMalloc((void**)&d_v,			NX * NY * sizeof(float)));
//    float *d_p;		    gpuErrchk(cudaMalloc((void**)&d_p,			NX * NY * sizeof(float)));
//
//    // copy host memory to device memory
//    gpuErrchk(cudaMemcpy(d_u, u,	 NX * NY * sizeof(float), cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_v, v,	 NX * NY * sizeof(float), cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(d_p, p,	 NX * NY * sizeof(float), cudaMemcpyHostToDevice));
//
//
//    // Grid size
//    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
//    dim3 dimGrid (iDivUp(NX, BLOCK_SIZE_X), iDivUp(NY, BLOCK_SIZE_Y));
//
//    project_kernel_3 <<< dimGrid, dimBlock >>> (d_u,d_v,d_p, NX, NY, h);
//
//
//    // --- Copy results from device to host
//    gpuErrchk(cudaMemcpy(u,d_u,	  NX * NY * sizeof(float), cudaMemcpyDeviceToHost));
//    gpuErrchk(cudaMemcpy(v,d_v,	  NX * NY * sizeof(float), cudaMemcpyDeviceToHost));
//
//    // free device memory
//    gpuErrchk(cudaFree(d_u));
//    gpuErrchk(cudaFree(d_v));
//    gpuErrchk(cudaFree(d_p));
//}


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


__global__ void set_bnd_kernel(int b, float * x, const int NX, const int NY, bool * occ, float * dens, float * u)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x ;
    const int j = blockIdx.y * blockDim.y + threadIdx.y ;


    // --- Only update "interior" (not boundary) node points
    if (i>0 && i<NX-1 && j>0 && j<NY-1)
    {
        // define boundary values for velocity and density

        // left and right wall
        if (b == 0) // density
        {
            x[get_idx(0,i,NX)] = x[get_idx(1,i,NX)]; // left
            x[get_idx(NX-1,i,NX)] = x[get_idx(NX-2,i,NX)];// right
        }
        if (b == 1) // u velocity component
        {
            x[get_idx(0,i,NX)] = -x[get_idx(1,i,NX)];// left
            x[get_idx(NX-1,i,NX)] = -x[get_idx(NX-2,i,NX)];// right
        }

        if (b == 2) // v velocity component
        {
            x[get_idx(0,i,NX)] = x[get_idx(1,i,NX)]; // left
            x[get_idx(NX-1,i,NX)] = x[get_idx(NX-2,i,NX)]; // right
        }

        // upper and lower wall

        if (b == 0) // density
        {
            x[get_idx(i,0 ,NX)] = x[get_idx(i,1,NX)];// bottom
            x[get_idx(i,NY-1,NX)] = x[get_idx(i,NY-2,NX)]; // top
        }

        if (b == 1) // u velocity component
        {
            x[get_idx(i,0 ,NX)] = x[get_idx(i,1,NX)];// bottom
            x[get_idx(i,NY-1,NX)] = x[get_idx(i,NY-2,NX)];// top
        }

        if (b == 2) // v velocity component
        {
            x[get_idx(i,0 ,NX)] = -x[get_idx(i,1,NX)]; // bottom
            x[get_idx(i,NY-1,NX)] = -x[get_idx(i,NY-2,NX)];// top
        }


        // implementing internal flow obstacles
        if(b != 0) {  // only changed boundaries for flow -> b = 1,2

            bool o = occ[get_idx(i,j, NX)];
            if(o == 1){
                x[get_idx(i-1,j,NX)] = b==1 ? -x[get_idx(i,j,NX)] : x[get_idx(i,j,NX)];
                x[get_idx(i+1,j,NX)] = b==1 ? -x[get_idx(i,j,NX)] : x[get_idx(i,j,NX)];
                x[get_idx(i,j-1,NX)] = b==2 ? -x[get_idx(i,j,NX)] : x[get_idx(i,j,NX)];
                x[get_idx(i,j+1,NX)] = b==2 ? -x[get_idx(i,j,NX)] : x[get_idx(i,j,NX)];
            }

        }


        // additional boundary conditions:

        if ((i > 0.4*NY && i < 0.5*NY)) {
            dens[get_idx(1, i,NX)] = 1.0;
            u[get_idx(1, i,NX)] = 10.0;

            dens[get_idx(i, 1,NX)] = 0.6;
            u[get_idx(i, 1,NX)] = 10.0;
        }

        // define edge cells as median of neighborhood
        x[get_idx(0 ,0 ,NX)]        = 0.5f*(x[get_idx(1,0,NX )]     + x[get_idx(0 ,1,NX)]);
        x[get_idx(0 ,NY-1,NX)]      = 0.5f*(x[get_idx(1,NY-1,NX)]   + x[get_idx(0 ,NY-2,NX)]);
        x[get_idx(NX-1,0,NX )]      = 0.5f*(x[get_idx(NX-2,0 ,NX)]  + x[get_idx(NX-1,1,NX)]);
        x[get_idx(NX-1,NY-1,NX)]    = 0.5f*(x[get_idx(NX-2,NY-1,NX)]+ x[get_idx(NX-1,NY-2,NX)]);
    }
}



//
//
//void try_set_bnd(int b, float * x, const int width, const int height, bool * occ, float * dens, float * u)
//{
//    const int NX = width+2;			    // --- Number of discretization points along the x axis
//    const int NY = height+2;			// --- Number of discretization points along the y axis
//
//    // allocate cuda memory
//    float *d_x;	        gpuErrchk(cudaMalloc((void**)&d_x,		    NX * NY * sizeof(float)));
//
//    // copy host memory to device memory
//    gpuErrchk(cudaMemcpy(d_x, x,	 NX * NY * sizeof(float), cudaMemcpyHostToDevice));
//
//
//    // Grid size
//    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
//    dim3 dimGrid (iDivUp(NX, BLOCK_SIZE_X), iDivUp(NY, BLOCK_SIZE_Y));
//
//    set_bnd_kernel <<< dimGrid, dimBlock >>> (d_x, NX, NY, b, occ, dens, u);
//
//
//    // --- Copy results from device to host
//    gpuErrchk(cudaMemcpy(x, d_x, NX * NY * sizeof(float), cudaMemcpyDeviceToHost));
//
//    // free device memory
//    gpuErrchk(cudaFree(d_x));
//}


void call_set_bnd_kernel(int b, float * d_x, const int NX, const int NY, dim3 dimGrid, dim3 dimBlock, bool * occ, float * dens, float * u)
{
    set_bnd_kernel <<< dimGrid, dimBlock >>> (b, d_x, NX, NY, occ, dens, u);
}




void call_add_source_kernel(float * d_x, float * d_x0, const int NX, const int NY, const float dt, dim3 dimBlock, dim3 dimGrid)
{
    add_source_kernel<<<dimGrid, dimBlock>>>(d_x,d_x0, NX, NY,dt);
}



void call_diffuse_kernel(float *x_d, float *x_prev_d, int NX, int NY, float a , dim3 dimBlock, dim3 dimGrid)
{
    diffuse_kernel <<< dimGrid, dimBlock >>> (x_d, x_prev_d, NX, NY, a);
}



void call_advect_kernel(float * d_x, float * d_x0, float * d_u,float * d_v, int NX, int NY, float dt, bool * occ, dim3 dimBlock, dim3 dimGrid)
{
    advect_kernel<<<dimGrid, dimBlock>>>(d_x, d_x0, d_u, d_v,  NX,  NY,  dt,  occ);
}

//
//void call_project_kernel_1(float * d_div,float * d_u,float * d_v,float * d_p, int NX, int NY, float h, dim3 dimBlock, dim3 dimGrid)
//{
//    add_source_kernel<<<dimGrid, dimBlock>>>(d_x,d_x0, NX, NY,dt);
//}
//
//
//void call_project_kernel_2(float * d_div,float * d_p, const int NX, const int NY, dim3 dimBlock, dim3 dimGrid)
//{
//    add_source_kernel<<<dimGrid, dimBlock>>>(d_x,d_x0, NX, NY,dt);
//}
//
//
//void call_project_kernel_3(float * d_d,float * d_d0,float * d_u,float * d_v, int NX, int NY, float h, dim3 dimBlock, dim3 dimGrid)
//{
//    add_source_kernel<<<dimGrid, dimBlock>>>(d_x,d_x0, NX, NY,dt);
//}
