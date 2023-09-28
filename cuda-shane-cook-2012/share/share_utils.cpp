#include "share.h"

static const char *getstr_cores_per_sm(int ccpmajor, int ccpminor, int SM_count)
{
	struct CPSmap_st 
	{
		int ccpmajor, ccpminor, cores;
	};

	static const CPSmap_st s_map[] = 
	{
		{ 2,0, 32 },
		{ 2,1, 48 },
		{ 3,0, 192 },
		{ 3,5, 192 },
		{ 5,0, 128 },
		{ 5,2, 128 },
		{ 6,0, 64 },
		{ 6,1, 128 },
		{ 6,2, 128 },
		{ 7,0, 64 },
		{ 7,5, 64 },
		{ 8,0, 64 },
		{ 8,6, 128 },
		{ 8,9, 128 },
		{ 9,0, 128 },
	};

	static char s_buf[40] = {};

	int cores = 0;

	for(int i=0; i<sizeof(s_map)/sizeof(s_map[0]); i++)
	{
		if(ccpmajor==s_map[i].ccpmajor && ccpminor==s_map[i].ccpminor)
		{
			cores = s_map[i].cores;
			break;
		}
	}

	if(cores>0)
		C_SNPRINTF(s_buf, sizeof(s_buf)-1, "%d (total %d)", cores, SM_count*cores);
	else
		C_SNPRINTF(s_buf, sizeof(s_buf)-1, "%s", "Unknown");

	return s_buf;
}

void myPrintGpuInfo()
{
	cudaDeviceProp  prop = {};

    int count = 0;
    cudaError_t cuerr = cudaGetDeviceCount( &count );
	if(count==0)
	{
		printf("No NVIDIA GPU found. This program cannot continue.\n");
		exit(2);
	}
	HANDLE_ERROR(cuerr);
	
	HANDLE_ERROR( cudaGetDeviceProperties( &prop, 0 ) );

	char gpustr[120] = {};
	printf("GPU: %s\nCompute-capability:%d.%d, SM-count:%d Cores-per-SM:%s\n",
		prop.name,
		prop.major, prop.minor,
		prop.multiProcessorCount,
		getstr_cores_per_sm(prop.major, prop.minor, prop.multiProcessorCount)
		);
	printf("\n");
}
