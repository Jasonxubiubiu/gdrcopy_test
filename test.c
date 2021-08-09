#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>

#define SIZE    (10*1024*1024)

float cuda_malloc_test( int size, bool up ) {
    cudaEvent_t start, stop;
    int *a, *dev_a;
    int i=0;
    float elapsedTime;
    cudaEventCreate( &start ) ;
    cudaEventCreate( &stop ) ;
    a = (int*)malloc( size * sizeof( *a ) );

        cudaMalloc( (void**)&dev_a, size * sizeof( *dev_a ) );
    cudaEventRecord( start, 0 );
    for (i=0; i<100; i++) {
    if (up)
        cudaMemcpy( dev_a, a,size * sizeof( *dev_a ),cudaMemcpyHostToDevice );
    else
        cudaMemcpy( a, dev_a,size * sizeof( *dev_a ),cudaMemcpyDeviceToHost ) ;
    }
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime,start, stop );
    free( a );
    cudaFree( dev_a );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return elapsedTime;
}

float cuda_host_alloc_test( int size, bool up ) {
    cudaEvent_t start, stop;
    int *a, *dev_a;
    float elapsedTime;
    int i=0;
    //printf("The dev_a is %");
    cudaEventCreate( &start ) ;
    cudaEventCreate( &stop ) ;
    cudaHostAlloc( (void**)&a,size * sizeof( *a ),cudaHostAllocDefault );
    cudaMalloc( (void**)&dev_a,size * sizeof( *dev_a ) );
    cudaEventRecord( start, 0 );
    for (i=0; i<100; i++) {
        if (up)
        cudaMemcpy( dev_a, a,size * sizeof( *a ),cudaMemcpyHostToDevice );
        else
        cudaMemcpy( a, dev_a,size * sizeof( *a ),cudaMemcpyDeviceToHost );
    }
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime,start, stop ) ;
    cudaFreeHost( a );
    cudaFree( dev_a );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    return elapsedTime;
}

int main( void ) {
    float elapsedTime;
    float MB = (float)100*SIZE*sizeof(int)/1024/1024;
    
    elapsedTime = cuda_malloc_test( SIZE, true );
    printf( "Time using cudaMalloc:%3.1f ms\n",elapsedTime );
    printf( "\tMB/s during copy up:%3.1f\n",MB/(elapsedTime/1000) );
    
    elapsedTime = cuda_malloc_test( SIZE, false );
    printf( "Time using cudaMalloc:%3.1f ms\n",elapsedTime );
    printf( "\tMB/s during copy down:%3.1f\n",MB/(elapsedTime/1000) );
    
    elapsedTime = cuda_host_alloc_test( SIZE, true );
    printf( "Time using cudaHostAlloc:%3.1f ms\n",elapsedTime );
    printf( "\tMB/s during copy up:%3.1f\n",MB/(elapsedTime/1000) );
    
    elapsedTime = cuda_host_alloc_test( SIZE, false );
    printf( "Time using cudaHostAlloc:%3.1f ms\n",elapsedTime );
    printf( "\tMB/s during copy down:%3.1f\n",MB/(elapsedTime/1000) );
}
