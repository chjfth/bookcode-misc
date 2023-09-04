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
#define SUBDIM_MAX 64
int SubDim = 16;

__global__ void kernel( unsigned char *ptr, bool need_sync, int subdim) 
{
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    __shared__ float    shared[SUBDIM_MAX][SUBDIM_MAX]; // cannot exceed 0xC000 bytes

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
    ptr[offset*4 + 1] = shared[(subdim-1)-threadIdx.x][(subdim-1)-threadIdx.y];
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
		printf("    unbalanced_syncthreads 0 16\n");
		printf("    unbalanced_syncthreads 0 32\n");
		printf("");
		printf("To run with __threadsync(), and see correct image, type:\n");
		printf("    unbalanced_syncthreads 1\n");
		printf("    unbalanced_syncthreads 1 16\n");
		printf("    unbalanced_syncthreads 1 8\n");
		printf("    unbalanced_syncthreads 1 4\n");
		return 1;
	}

	bool need_sync = (argv[1][0]=='1') ? true : false;

	if(argc>2)
		SubDim = (int)strtoul(argv[2], nullptr, 0);

    DataBlock   data;
    CPUBitmap bitmap( DIM, DIM, &data );
    unsigned char    *dev_bitmap;

    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap,
                              bitmap.image_size() ) );
    data.dev_bitmap = dev_bitmap;

    dim3    grids(DIM/SubDim, DIM/SubDim);
    dim3    threads(SubDim, SubDim);
    kernel<<<grids,threads>>>( dev_bitmap, need_sync, SubDim );

    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost ) );
                              
    HANDLE_ERROR( cudaFree( dev_bitmap ) );
                              
    bitmap.display_and_exit();
}


