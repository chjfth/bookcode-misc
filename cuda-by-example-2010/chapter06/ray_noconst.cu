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

#include "ray_share.h" // Chj refactors this


int main( int argc, char *argv[] ) 
{
	if(argc==1) {
		printf("Hint: You can pass a integer parameter as random seed.\n");
	}
	else {
		unsigned seed = strtoul(argv[1], nullptr, 0);
		printf("Info: Use random seed %u.\n", seed);
		srand(seed);
	}

	// capture the start time
	cudaEvent_t     start, stop;
	HANDLE_ERROR( cudaEventCreate( &start ) );
	HANDLE_ERROR( cudaEventCreate( &stop ) );
	HANDLE_ERROR( cudaEventRecord( start, 0 ) );

	CPUBitmap bitmap( DIM, DIM );
	unsigned char   *dev_bitmap;

	// allocate memory on the GPU for the output bitmap
	HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap,
		bitmap.image_size() ) );
	// allocate memory for the Sphere dataset
	HANDLE_ERROR( cudaMalloc( (void**)&s,
		sizeof(Sphere) * SPHERES ) );

	// allocate temp memory, initialize it, copy to
	// memory on the GPU, then free our temp memory
	Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );
	for (int i=0; i<SPHERES; i++) {
		temp_s[i].r = rnd( 1.0f );
		temp_s[i].g = rnd( 1.0f );
		temp_s[i].b = rnd( 1.0f );
		temp_s[i].x = rnd( 1000.0f ) - 500;
		temp_s[i].y = rnd( 1000.0f ) - 500;
		temp_s[i].z = rnd( 1000.0f ) - 500;
		temp_s[i].radius = rnd( 100.0f ) + 20;
	}
	HANDLE_ERROR( cudaMemcpy( s, temp_s,
		sizeof(Sphere) * SPHERES,
		cudaMemcpyHostToDevice ) );
	free( temp_s );

	// generate a bitmap from our sphere data
	dim3    grids(DIM/16,DIM/16);
	dim3    threads(16,16);
	kernel<<<grids,threads>>>( s, dev_bitmap );

	// copy our bitmap back from the GPU for display
	HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
		bitmap.image_size(),
		cudaMemcpyDeviceToHost ) );

	// get stop time, and display the timing results
	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );
	float   elapsedTime;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
	printf( "Time to generate:  %3.1f ms\n", elapsedTime );

	HANDLE_ERROR( cudaEventDestroy( start ) );
	HANDLE_ERROR( cudaEventDestroy( stop ) );

	HANDLE_ERROR( cudaFree( dev_bitmap ) );
	HANDLE_ERROR( cudaFree( s ) );

	// display
	bitmap.display_and_exit();
}
