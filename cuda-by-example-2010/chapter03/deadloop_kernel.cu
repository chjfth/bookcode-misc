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

__global__ void kernel( void ) 
{
	int temp = 0;
	while(1)
		temp++;
}

int main( void ) 
{
	printf( "Kernel dead loop start.\n" );

	uint64 usec_start = ps_GetOsMicrosecs64();
	kernel<<<1,1>>>();
	uint64 usec_end = ps_GetOsMicrosecs64();

	printf( "Kernel dead loop end. (%s millisec)\n", 
		us_to_msecstring(usec_end - usec_start));
    return 0;
}
