#include <stdio.h>
#include <stdlib.h>

// Chj: 
// This program does experiment as p89 suggests: we call __syncthreads()
// for odd threads only, but NOT for even threads. Let's see whether
// the "odd"(pun) __syncthreads() would freeze/hang.
//
// The code body is adapted from add_loop_gpu.cu .

#include "../common/book.h"

#define N   10

__global__ void add( int *a, int *b, int *c, bool is_half_sync ) 
{
	int tid = threadIdx.x;    // this thread handles the data at its thread id
	if (tid < N)
		c[tid] = a[tid] + b[tid];

	// Experiment for unbalanced __syncthreads()
	if( is_half_sync && (threadIdx.x % 2 == 1) )
	{
		__syncthreads();
	}
}

int main( int argc, char *argv[] ) 
{
	if(argc==1)
	{
		printf("To run normally, type:\n");
		printf("    unbalanced_syncthreads 0\n");
		printf("");
		printf("To run normally, type:\n");
		printf("    unbalanced_syncthreads 1\n");
		return 1;
	}

	bool is_half_sync = (argv[1][0]=='1') ? true : false;

	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	// allocate the memory on the GPU
	HANDLE_ERROR( cudaMalloc( (void**)&dev_a, N * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_b, N * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_c, N * sizeof(int) ) );

	// fill the arrays 'a' and 'b' on the CPU
	for (int i=0; i<N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	// copy the arrays 'a' and 'b' to the GPU
	HANDLE_ERROR( cudaMemcpy( dev_a, a, N * sizeof(int),
		cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_b, b, N * sizeof(int),
		cudaMemcpyHostToDevice ) );

	add<<<1, N>>>( dev_a, dev_b, dev_c, is_half_sync );

	// copy the array 'c' back from the GPU to the CPU
	HANDLE_ERROR( cudaMemcpy( c, dev_c, N * sizeof(int),
		cudaMemcpyDeviceToHost ) );

	// display the results
	for (int i=0; i<N; i++) {
		printf( ">> %d + %d = %d\n", a[i], b[i], c[i] );
	}

	// free the memory allocated on the GPU
	HANDLE_ERROR( cudaFree( dev_a ) );
	HANDLE_ERROR( cudaFree( dev_b ) );
	HANDLE_ERROR( cudaFree( dev_c ) );

	return 0;
}
