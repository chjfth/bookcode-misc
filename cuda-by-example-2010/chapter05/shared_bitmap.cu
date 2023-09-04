/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 512
#define PI 3.1415926535897932f
#define SUBDIM 32

__global__ void kernel( unsigned char *ptr, bool need_sync) 
{
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    __shared__ float    shared[SUBDIM][SUBDIM];

    // now calculate the value at that position
    const float period = 128.0f;

    shared[threadIdx.x][threadIdx.y] =
            255 * (sinf(x*2.0f*PI/ period) + 1.0f) *
                  (sinf(y*2.0f*PI/ period) + 1.0f) / 4.0f;

    // removing this syncthreads shows graphically what happens
    // when it doesn't exist.  this is an example of why we need it.
	if(need_sync) {
		__syncthreads();
	}

    ptr[offset*4 + 0] = 0;
    ptr[offset*4 + 1] = shared[(SUBDIM-1)-threadIdx.x][(SUBDIM-1)-threadIdx.y];
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

int main( int argc, char *argv[] ) 
{
	if(argc==1)
	{
		printf("To run without __threadsync(), and see garbled image, type:\n");
		printf("    unbalanced_syncthreads 0\n");
		printf("");
		printf("To run with __threadsync(), and see correct image, type:\n");
		printf("    unbalanced_syncthreads 1\n");
		return 1;
	}

	bool need_sync = (argv[1][0]=='1') ? true : false;
    DataBlock   data;
    CPUBitmap bitmap( DIM, DIM, &data );
    unsigned char    *dev_bitmap;

    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap,
                              bitmap.image_size() ) );
    data.dev_bitmap = dev_bitmap;

    dim3    grids(DIM/SUBDIM, DIM/SUBDIM);
    dim3    threads(SUBDIM, SUBDIM);
    kernel<<<grids,threads>>>( dev_bitmap, need_sync );

    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost ) );
                              
    HANDLE_ERROR( cudaFree( dev_bitmap ) );
                              
    bitmap.display_and_exit();
}


