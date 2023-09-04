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

__global__ void kernel( void ) 
{
	int temp = 0;
	while(1)
		temp++;
}

int main( void ) 
{
	printf( "Dead loop start.\n" );

	kernel<<<1,1>>>();

	printf( "Dead loop end.\n" );
    return 0;
}
