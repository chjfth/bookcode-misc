#ifndef __share_h_
#define __share_h_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

typedef unsigned char Uchar;
typedef unsigned int Uint;

#define OCC_DIVIDE(n, x) ( ((n)+(x)-1) / (x) ) // occupation divide


static cudaError_t PrintCudaError(cudaError_t err, const char *file, int line)
{
	if(err)
	{
		printf( "[CUDAERR:%d] %s in %s at line %d\n", 
			err,
			cudaGetErrorString( err ),
			file, line 
			);
	}
	return err;
}

#define PRINT_ERROR(err) (PrintCudaError(err, __FILE__, __LINE__))

static void HandleError( cudaError_t err,
	const char *file,
	int line ) 
{
	if (err != cudaSuccess) {
		PrintCudaError(err, file, line);
		exit( EXIT_FAILURE );
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
	printf( "Host memory failed in %s at line %d\n", \
	__FILE__, __LINE__ ); \
	exit( EXIT_FAILURE );}}


#endif
