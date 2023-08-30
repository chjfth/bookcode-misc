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


#include "../common/book.h"

void chj_bad_gpumem();

__global__ void add( int a, int b, int *c ) {
    *c = a + b;
}

int main( void ) {
    int c;
    int *dev_c;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c, sizeof(int) ) );

    add<<<1,1>>>( 2, 7, dev_c );

    HANDLE_ERROR( cudaMemcpy( &c, dev_c, sizeof(int), cudaMemcpyDeviceToHost ) );
    printf( "2 + 7 = %d\n", c );

	chj_bad_gpumem();

	add<<<1,1>>>( 3, 8, dev_c );
	HANDLE_ERROR( cudaMemcpy( &c, dev_c, sizeof(int), cudaMemcpyDeviceToHost ) );
	printf( "3 + 8 = %d\n", c );
	
	HANDLE_ERROR( cudaFree( dev_c ) );

    return 0;
}

void chj_bad_gpumem()
{
	int *bad_addrs[] = { (int*)0, (int*)0x1000, (int*)0x40000, (int*)0xCC0000 };
	for(int i=0; i<ARRAYSIZE(bad_addrs); i++)
	{
		int *bad_addr = bad_addrs[i];
		add<<<1,1>>>(1, 5, bad_addr);

		printf("Tried bad addr: %p\n", bad_addr);
	}
}
