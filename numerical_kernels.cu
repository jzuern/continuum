////
//// Created by jannik on 22/05/18.
////
//
#include <cuda_runtime.h>
#include <cuda.h>
#include "numerical_kernels.h"

#define TILE_WIDTH 2

__global__ void heat_kernel( float *data_array_new_device , float *data_array_old_device , const int width )
{

    // calculate thread id
    printf("hello from thread %i\n",threadIdx.x );
    unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
    unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;

    data_array_new_device[row*width + col] = (data_array_old_device[row * width + k ] + 10) % 256;

}



void launch_kernel()
{

    //copy host array to device array; cudaMemcpy ( dest , source , WIDTH , direction )
    cudaMemcpy ( data_array_new_device , data_array_new_host , height*width*sizeof (float) , cudaMemcpyHostToDevice ) ;
    cudaMemcpy ( data_array_old_device, data_array_old_host , height*width*sizeof (float) , cudaMemcpyHostToDevice ) ;

    //calling kernal
    dim3 dimGrid ( width/TILE_WIDTH , height/TILE_WIDTH ,1 ) ;
    dim3 dimBlock( TILE_WIDTH, TILE_WIDTH, 1 ) ;


    heat_kernel <<<dimGrid, dimBlock>>> ( data_array_new_device , data_array_old_device , width) ;

    // copy back resutls
    cudaMemcpy(data_array_new_host , data_array_new_device , height*width*sizeof(float), cudaMemcpyDeviceToHost) ;

}

void cuda_init()
{


    // populate cuda stuff
    for (int i = 0 ; i<width ; i++ )
    {
        for (int j = 0 ; j<height ; j++ )
        {
            data_array_new_host[i][j] = 1.;
        }
    }

    //create device array cudaMalloc ( (void **)&array_name, sizeofmatrixinbytes) ;
    cudaMalloc((void **) &data_array_new_device , height*width*sizeof (float) ) ;
}



// TESTING



__global__ void MatrixMul( float *Md , float *Nd , float *Pd , const int WIDTH )
{

    // calculate thread id
    printf("hello from thread %i\n",threadIdx.x );
    unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
    unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;

    for (int k = 0 ; k<WIDTH ; k++ )
    {
        Pd[row*WIDTH + col]+= Md[row * WIDTH + k ] * Nd[ k * WIDTH + col] ;

    }
}

void try_cuda()
{


    const int WIDTH = 100 ;
    float array1_h[WIDTH][WIDTH] ,array2_h[WIDTH][WIDTH],
            result_array_h[WIDTH][WIDTH] ,M_result_array_h[WIDTH][WIDTH]  ;

    float *array1_d , *array2_d ,*result_array_d  ,*M_result_array_d ; // device array
    int i , j ;
    //input in host array
    for ( i = 0 ; i<WIDTH ; i++ )
    {
        for (j = 0 ; j<WIDTH ; j++ )
        {
            array1_h[i][j] = 1. ;
            array2_h[i][j] = 2. ;
        }
    }

    //create device array cudaMalloc ( (void **)&array_name, sizeofmatrixinbytes) ;
    cudaMalloc((void **) &array1_d , WIDTH*WIDTH*sizeof (int) ) ;
    cudaMalloc((void **) &array2_d , WIDTH*WIDTH*sizeof (int) ) ;

    //copy host array to device array; cudaMemcpy ( dest , source , WIDTH , direction )
    cudaMemcpy ( array1_d , array1_h , WIDTH*WIDTH*sizeof (int) , cudaMemcpyHostToDevice ) ;
    cudaMemcpy ( array2_d , array2_h , WIDTH*WIDTH*sizeof (int) , cudaMemcpyHostToDevice ) ;

    //allocating memory for resultent device array
    cudaMalloc((void **) &result_array_d , WIDTH*WIDTH*sizeof (int) ) ;
    cudaMalloc((void **) &M_result_array_d , WIDTH*WIDTH*sizeof (int) ) ;

    //calling kernel
    dim3 dimGrid ( WIDTH/TILE_WIDTH , WIDTH/TILE_WIDTH ,1 ) ;
    dim3 dimBlock( TILE_WIDTH, TILE_WIDTH, 1 ) ;


    MatrixMul <<<dimGrid,dimBlock>>> ( array1_d , array2_d ,M_result_array_d , WIDTH) ;


    // all gpu function blocked till kernel is working
    //copy back result_array_d to result_array_h

    cudaMemcpy(M_result_array_h , M_result_array_d , WIDTH*WIDTH*sizeof(int) ,
               cudaMemcpyDeviceToHost) ;

    //printf the result array
    for ( i = 0 ; i<WIDTH ; i++ )
    {
        for ( j = 0 ; j < WIDTH ; j++ )
        {
            printf ("%f   ",M_result_array_h[i][j] ) ;
        }
        printf ("\n") ;
    }

}




void test()
{
    printf("test \n");
}