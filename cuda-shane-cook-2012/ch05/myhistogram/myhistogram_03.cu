#include "mykernels.h"


__shared__ unsigned int d_bin_data_shared[BIN256];

__global__ void myhistogram_03a( // @page 101 modified
	const unsigned int * d_hist_data,
	unsigned int * d_bin_data,
	int sample_ints)
{
	// Feature: Partial counting into shared-mem, then merge the result to global-mem.
	// Chj: Note: this program implies threadIdx.y==1
	// Each call copes with four user samples(each sample is one byte).

	/* Work out our thread id */
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	Uint tid = idx + idy * blockDim.x * gridDim.x;
	Uint threads_per_block = blockDim.x * blockDim.y;
	
	// Chj: Clear the d_bin_data_shared[] array.
	int idxBin = threadIdx.x;
	while(idxBin < BIN256)
	{
		d_bin_data_shared[idxBin] = 0;
		idxBin += threads_per_block;
	}

	// All threads should wait for the above clearing done.
	__syncthreads();

	// Partial counting into d_bin_data_shared[]
	//
	if(tid < sample_ints)
	{
		// Fetch four histogram elements in a group.
		Uint value_u32 = d_hist_data[tid];

		atomicAdd( &(d_bin_data_shared[ (value_u32 & 0x000000FF) ]), 1 );
		atomicAdd( &(d_bin_data_shared[ (value_u32 & 0x0000FF00) >>  8 ]), 1 );
		atomicAdd( &(d_bin_data_shared[ (value_u32 & 0x00FF0000) >> 16 ]), 1 );
		atomicAdd( &(d_bin_data_shared[ (value_u32 & 0xFF000000) >> 24 ]), 1 );
	}
	
	/* Wait for all threads to update shared memory, again */
	__syncthreads();

#if 0
	// Chj: Let the first thread accumulate the counting result. 
	// ( This slows down the overall myhistogram_03a() by 5~10X. Very bad.)
	if(threadIdx.x==0)
	{
		for(int i=0; i<BIN256; i++)
		{
			atomicAdd( &d_bin_data[i], d_bin_data_shared[i] );
		}
	}
#else
	// Chj: This is much better, all threads in current block are utilized.
	// On my RTX 3050 card, with GeForce driver 536.40, 
	// running `myhistogram 1024000 512`, myhistogram_03a is 20X the speed of myhistogram_02.
	//
	// But, on my GTX 870M, with GeForce driver 376.54, 
	// running `myhistogram 1024000 512`, myhistogram_03a is merely 20% faster than myhistogram_02.
	idxBin = threadIdx.x;
	while( idxBin < BIN256 )
	{
		atomicAdd( &d_bin_data[idxBin], d_bin_data_shared[idxBin]);
		idxBin += threads_per_block;
	}
#endif
}

__global__ void myhistogram_03b( // byte-by-byte operation based on myhistogram_03a
	const Uchar * d_hist_data,
	Uint * const d_bin_data,
	int sample_count)
{
	// Chj extra: Each call copes only ONE sample byte.
	// On RTX 3050, driver 536.40, running `myhistogram.exe 10240000 512`,
	// This exhibits only 25% ~ 30% speed of myhistogram_03b.
	// -- Perhaps it's due to myhistogram_03a has fewer total contending threads
	//    (25% of myhistogram_03b).

	/* Work out our thread id */
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	Uint tid = idx + idy * blockDim.x * gridDim.x;
	Uint threads_per_block = blockDim.x * blockDim.y;

	// Chj: Clear the d_bin_data_shared[] array.
	int idxBin = threadIdx.x;
	while(idxBin < BIN256)
	{
		d_bin_data_shared[idxBin] = 0;
		idxBin += threads_per_block;
	}

	// All threads should wait for the above clearing done.
	__syncthreads();

	// Partial counting into d_bin_data_shared[]
	//
	if(tid < sample_count)
	{
		Uchar who = d_hist_data[tid];
		atomicAdd( &d_bin_data_shared[who], 1 );
	}

	// All threads should wait for other-thread's histogram counting done.
	__syncthreads();

	idxBin = threadIdx.x;
	while( idxBin < BIN256 )
	{
		atomicAdd( &d_bin_data[idxBin], d_bin_data_shared[idxBin]);
		idxBin += threads_per_block;
	}
}

