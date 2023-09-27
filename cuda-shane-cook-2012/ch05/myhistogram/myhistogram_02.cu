#include "mykernels.h"


__global__ void myhistogram_02( // @page 99-100
	const unsigned int * d_hist_data, // note: each call will process 4 bytes(Uint)
	unsigned int * d_bin_data,
	int sample_ints) 
{
	/* Work out our thread id */
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int tid = idx + idy * blockDim.x * gridDim.x;

	if(tid<sample_ints)
	{
		/* Fetch the data value */
		const Uint value_u32 = d_hist_data[tid];

		atomicAdd( &(d_bin_data[ (value_u32 & 0x000000FF) ]), 1 );
		atomicAdd( &(d_bin_data[ (value_u32 & 0x0000FF00) >>  8 ]), 1 );
		atomicAdd( &(d_bin_data[ (value_u32 & 0x00FF0000) >> 16 ]), 1 );
		atomicAdd( &(d_bin_data[ (value_u32 & 0xFF000000) >> 24 ]), 1 );
	}
}
