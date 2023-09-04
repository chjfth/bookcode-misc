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
#include "../common/chjdbg.h"

__global__ void kernel_null_work( int n ) {}

__global__ void kernel_dead_loop( int n ) 
{
	int temp = n;
	while(1)
		temp++;
}

int main( void ) 
{
	printf( "Kernel dead loop start.\n" );

	uint64 usec0 = ps_GetOsMicrosecs64();

	kernel_null_work<<<1,1>>>(1);

	uint64 usec1 = ps_GetOsMicrosecs64();

	kernel_null_work<<<1,1>>>(2);

	uint64 usec2 = ps_GetOsMicrosecs64();

	kernel_dead_loop<<<1,1>>>(0);

	uint64 usec3 = ps_GetOsMicrosecs64();

	printf( "kernel_null_work(1) : %s millisec)\n", us_to_msecstring(usec1-usec0) );
	printf( "kernel_null_work(2) : %s millisec)\n", us_to_msecstring(usec2-usec1) );
	printf( "kernel_dead_loop(0) : %s millisec)\n", us_to_msecstring(usec3-usec2) );
    return 0;
}
