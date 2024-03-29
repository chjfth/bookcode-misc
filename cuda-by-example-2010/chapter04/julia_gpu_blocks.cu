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

// [2023-09-02] Chj: User needs to assign parameters <dim> and <scale> .

#include "../common/book.h"
#include "../common/cpu_bitmap.h"
#include "../common/chjdbg.h"

int dim = 1000;
float scale = 1.5;

struct cuComplex 
{
    float   r;
    float   i;
    __device__ cuComplex( float a, float b ) : r(a), i(b)  {}
    __device__ float magnitude2( void ) {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__ int julia( int x, int y, int dim, float scale ) 
{
    float jx = scale * (float)(dim/2 - x)/(dim/2);
    float jy = scale * (float)(dim/2 - y)/(dim/2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

__global__ void kernel( unsigned char *ptr, int dim, float scale ) 
{
    // map from blockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x; // assert(dim==gridDim.x)

    // now calculate the value at that position
    int juliaValue = julia( x, y, dim, scale );
    ptr[offset*4 + 0] = 255 * juliaValue;
    ptr[offset*4 + 1] = 0;
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;
}

int main( int argc, char *argv[] ) 
{
	if(argc<3) {
		printf("Need at least two parameters. \n");
		printf("    julia_gpu_blocks <sample_points> <scale> [sleep_millisec]\n");
		printf("\n");
		printf("For example:\n");
		printf("    julia_gpu_blocks 1000 1.5\n");
		printf("    julia_gpu_blocks 500  5.0 156\n");
		return 1;
	}

	dim = (int)strtoul(argv[1], nullptr, 0);
	scale = (float)atof(argv[2]);

	int sleep_millisec = 0;
	if(argc>3) {
		sleep_millisec = (int)strtoul(argv[3], nullptr, 0);
	}

	if(dim<=0) {
		printf("ERROR: <sample_points> must > 0, given: %d\n", dim);
		return 4;
	}

	if(scale<=0) {
		printf("ERROR: <scale> must > 0, given: %g\n", scale);
		return 4;
	}

	printf("Using sample_points=%d , scale=%g\n", dim, scale);
	if(sleep_millisec>0) {
		printf("Will sleep %d millisec after GPU kernel call.\n", sleep_millisec);
	}

    CPUBitmap bitmap( dim, dim );
    unsigned char    *dev_bitmap;

    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );

	unsigned int64 usec_start = ps_GetOsMicrosecs64(); // chj

    dim3    grid(dim, dim);
    kernel<<<grid,1>>>( dev_bitmap, dim, scale );
	
	unsigned int64 usec_done0 = ps_GetOsMicrosecs64(); // chj

	if(sleep_millisec>0) {
		// [2023-09-09] Chj experiment: If we sleep some seconds here, can we see 
		// subsequent cudaMemcpy() costs less time? No, I do not see time reduction 
		// between usec_done1 and usec_done2? No, I do not see time reduction on 
		// a GTX 870M, whether it is a Debug build or Release build.
		Sleep(sleep_millisec);
	}

	unsigned int64 usec_done1 = ps_GetOsMicrosecs64(); // chj

    cudaError_t cudaerr = PRINT_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost ) );

	unsigned int64 usec_done2 = ps_GetOsMicrosecs64(); // chj

	printf("Julia calculation time cost milliseconds (GPU): %s\n", 
		us_to_msecstring(usec_done0 - usec_start));
	
	if(cudaerr)
	{
		printf("cudaMemcpyDeviceToHost %d bytes, cost milliseconds: %s (FAIL!)\n", 
			bitmap.image_size(), us_to_msecstring(usec_done2 - usec_done1));
		exit(4);
	}
	else
	{
		printf("cudaMemcpyDeviceToHost %d bytes, cost milliseconds: %s\n", 
			bitmap.image_size(), us_to_msecstring(usec_done2 - usec_done1));
	}

    HANDLE_ERROR( cudaFree( dev_bitmap ) );
                              
    bitmap.display_and_exit();
}
