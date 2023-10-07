#include "mykernels.h"


__global__ void myhistogram_02N( // #02 with Nbatch param
	const Uint *d_hist_data, // note: each call will process 4 bytes(Uint)
	Uint *d_bin_data,
	int sample_ints,
	Uint Nbatch) 
{
	assert(blockDim.y==1);

	/* Work out our thread id */
	Uint idxN = (blockIdx.x * (blockDim.x*Nbatch)) + threadIdx.x;
	Uint idyN = (blockIdx.y * blockDim.y) + threadIdx.y;
	Uint tidN = idxN + idyN * (blockDim.x*Nbatch) * gridDim.x;

	Uint threads_per_block = blockDim.x * blockDim.y;

	for(Uint i=0, tid_offset=0; 
		i<Nbatch; 
		i++, tid_offset += threads_per_block)
	{
		Uint uint_offset = tidN + tid_offset;

		if(uint_offset < sample_ints)
		{
			/* Fetch the data value */
			const Uint value_u32 = d_hist_data[uint_offset];

			atomicAdd( &(d_bin_data[ (value_u32 & 0x000000FF) ]), 1 );
			atomicAdd( &(d_bin_data[ (value_u32 & 0x0000FF00) >>  8 ]), 1 );
			atomicAdd( &(d_bin_data[ (value_u32 & 0x00FF0000) >> 16 ]), 1 );
			atomicAdd( &(d_bin_data[ (value_u32 & 0xFF000000) >> 24 ]), 1 );
		}
	}
}
