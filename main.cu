#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>
#include <gtkmm.h>
#include "simulation.h"
#include <cuda.h>
//#include "square.cuh"

#include <math.h>

using namespace std;

#define TILE_WIDTH 2

/*matrix multiplication kernels*/

//non shared
__global__ void MatrixMul( float *Md , float *Nd , float *Pd , const int WIDTH )
{

    // calculate thread id
    printf("hello from thread %i\n",threadIdx.x );
    unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
    unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;

    for (int k = 0 ; k<WIDTH ; k++ )
    {
        Pd[row*WIDTH + col]+= Md[row * WIDTH + k ] * Nd[ k * WIDTH + col] ;
        Pd[row*WIDTH + col]+= Md[row * WIDTH + k ] + Nd[ row * WIDTH + k ] ;

    }
}

void try_cuda()
{
    const int WIDTH = 6 ;
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

    //calling kernal
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


int main(int argc, char* argv[])
{


//    try_cuda();


    Glib::RefPtr<Gtk::Application> app = Gtk::Application::create(argc, argv, "com.kaze.test");

    Simulation sim;

    // The Gui Window is displayed
    return app->run(sim);
}

