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
#include "../common/cpu_anim.h"
#include "../common/chjdbg.h"

const int PROBE_FRAMES = 200;
unsigned int gar_frame_rusec[PROBE_FRAMES]; // Nth-ele: relative-microseconds from animation frame 0.
uint64 g_frame0_usec = 0;
int g_frame_idx = 0;

int dim = 1024;
#define PI 3.1415926535897932f

__global__ void kernel( unsigned char *ptr, int ticks, int dim ) 
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	// Chj note1: If dim is not a multiple of 16, (blockDim.x*gridDim.x)!=dim .
	int offset = x + y * dim; // (blockDim.x * gridDim.x) => dim;

	if(x > dim)	{
		// Chj note2: avoid overwriting next scanline.
		return; 
	}

	// now calculate the value at that position
	float fx = x - dim/2;
	float fy = y - dim/2;
	float d = sqrtf( fx * fx + fy * fy );
	unsigned char grey = (unsigned char)
		(128.0f + 127.0f * cos(d/10.0f - ticks/7.0f) / (d/10.0f + 1.0f));    
	ptr[offset*4 + 0] = grey;
	ptr[offset*4 + 1] = grey;
	ptr[offset*4 + 2] = grey;
	ptr[offset*4 + 3] = 255;
}

struct DataBlock {
	unsigned char   *dev_bitmap;
	CPUAnimBitmap  *bitmap;
};

void generate_frame( DataBlock *d, int ticks ) 
{
	uint64 now_usec = ps_GetOsMicrosecs64();
	if(g_frame_idx==0)
		g_frame0_usec = now_usec;

	if(g_frame_idx<PROBE_FRAMES) {
		gar_frame_rusec[g_frame_idx++] = now_usec - g_frame0_usec;

		if(g_frame_idx==PROBE_FRAMES) {
			printf("Animation frame callback timing:\n");
			dump_microseconds_diffs(gar_frame_rusec, PROBE_FRAMES);
		}
	}

	dim3    blocks(OCC_DIVIDE(dim,16), OCC_DIVIDE(dim,16));
	dim3    threads(16, 16);

	kernel<<<blocks,threads>>>( d->dev_bitmap, ticks, dim );

	HANDLE_ERROR( cudaMemcpy( d->bitmap->get_ptr(),
		d->dev_bitmap,
		d->bitmap->image_size(),
		cudaMemcpyDeviceToHost ) );
}

// clean up memory allocated on the GPU
void cleanup( DataBlock *d ) 
{
	HANDLE_ERROR( cudaFree( d->dev_bitmap ) ); 
}



int main( int argc, char *argv[] ) 
{
	if(argc==1)
	{
		printf("Usage:\n");
		printf("    ripple2 <drawing_size>\n");
		printf("\n");
		printf("Exmple:\n");
		printf("    ripple2 1024\n");
		return 1;
	}

	dim = (int)strtoul(argv[1], nullptr, 0);
	if(dim<16)
	{
		printf("<drawing_size> must be at least 16.\n");
		return 1;
	}

	DataBlock   data;
	CPUAnimBitmap  bitmap( dim, dim, &data );
	data.bitmap = &bitmap;

	HANDLE_ERROR( cudaMalloc( (void**)&data.dev_bitmap,
		bitmap.image_size() ) );

	char title[40] = {};
	_snprintf_s(title, _TRUNCATE, "ripple %d", dim);

	bitmap.anim_and_exit( 
		(void (*)(void*,int))generate_frame,
		(void (*)(void*))cleanup,
		title);
}
