#include "mykernels.h"


__global__ void myhistogram_01( // @page 98
	const unsigned char * d_hist_data,
	unsigned int * d_bin_data,
	int sample_count) 
{
	/* Work out our thread id */
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int tid = idx + idy * blockDim.x * gridDim.x;

	if(tid<sample_count)
	{
		/* Fetch the data value */
		const unsigned char value = d_hist_data[tid];

//		printf("[#%d] .%u\n", tid, value);

		atomicAdd(&(d_bin_data[value]), 1);
	}
}
