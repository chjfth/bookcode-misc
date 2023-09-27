#include "mykernels.h"


__global__ void myhistogram_07( // @page 102
	const Uint * d_hist_data,
	Uint * d_bin_data,
	int sample_ints,
	Uint Nbatch) 
{
	// Chj: Note: this program still implies threadIdx.y==1

	/* Work out our thread id */
	Uint idx = (blockIdx.x * (blockDim.x*Nbatch)) + threadIdx.x;
	Uint idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	Uint tid = idx + idy * (blockDim.x*Nbatch) * gridDim.x;

	Uint idx_gpu = (blockIdx.x * blockDim.x) + threadIdx.x;
	Uint idy_gpu = (blockIdx.y * blockDim.y) + threadIdx.y;
	Uint tid_gpu = idx + idy * blockDim.x * gridDim.x;
	Uint tidQ = tid_gpu * Nbatch;
	
	Uint threads_per_block = blockDim.x * blockDim.y;

	if(tid!=tidQ)
	{
		printf("[b=%d,t=%d]Got bad.....(%d vs %d)\n", blockIdx.x, threadIdx.x,  tid, tidQ);
	}

	// Chj: Clear the d_bin_data_shared[] array.
	int idxBin = threadIdx.x;
	while(idxBin < BIN256)
	{
		d_bin_data_shared[idxBin] = 0;
		idxBin += threads_per_block;
	}

	// All threads should wait for the above clearing done.
	__syncthreads();

	// Fetch multiple histogram elements, and update poll-bin accordingly.
	//
	for(Uint i=0, tid_offset=0; 
		i<Nbatch; 
		i++, tid_offset += threads_per_block)
	{
		// note: tid_offset is counted in sizeof(Uint).

		Uint uint_offset = tidQ + tid_offset;

		if(uint_offset < sample_ints)
		{
			// Fetch four histogram elements in a group.
			Uint value_u32 = d_hist_data[uint_offset];

			atomicAdd( &(d_bin_data_shared[ (value_u32 & 0x000000FF) ]), 1 );
			atomicAdd( &(d_bin_data_shared[ (value_u32 & 0x0000FF00) >>  8 ]), 1 );
			atomicAdd( &(d_bin_data_shared[ (value_u32 & 0x00FF0000) >> 16 ]), 1 );
			atomicAdd( &(d_bin_data_shared[ (value_u32 & 0xFF000000) >> 24 ]), 1 );
		}
	}

	/* Wait for all threads to update shared memory, again */
	__syncthreads();

	// Merge block result in d_bin_data_shared[] into d_bin_data[]
	idxBin = threadIdx.x;
	while( idxBin < BIN256 )
	{
		atomicAdd( &d_bin_data[idxBin], d_bin_data_shared[idxBin]);
		idxBin += threads_per_block;
	}
}
