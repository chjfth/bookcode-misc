#include "share.h"

void myPrintGpuInfo()
{
	cudaDeviceProp  prop = {};

    int count = 0;
    HANDLE_ERROR( cudaGetDeviceCount( &count ) );

	if(count==0)
	{
		printf("No NVIDIA GPU found. The program cannot continue.\n");
		exit(2);
	}

	for (int i=0; i< count; i++) 
	{
        HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );

		printf( "   --- General Information for device %d ---\n", i );
        printf( "Name:  %s\n", prop.name );
        printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );

	}
}
