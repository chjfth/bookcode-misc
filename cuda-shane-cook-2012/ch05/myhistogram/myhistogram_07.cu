#include "mykernels.h"


__global__ void myhistogram_07( // @page 102
	const unsigned char * d_hist_data,
	unsigned int * d_bin_data,
	int sample_count,
	Uint Nbatch) 
{
	/* Work out our thread id */
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int tid = idx + idy * blockDim.x * gridDim.x;

	// TODO
}
