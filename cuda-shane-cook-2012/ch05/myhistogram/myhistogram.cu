#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "../../share/share.h"

#define BIN256 256
//#define THREADS256 256

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

__shared__ unsigned int d_bin_data_shared[BIN256];

__global__ void myhistogram_03a( // @page 101 modified
	const unsigned int * d_hist_data,
	unsigned int * const d_bin_data,
	int sample_ints)
{
	// Chj: Note: this program implies threadIdx.y==1
	// Each call copes with four user samples(each sample is one byte).

	/* Work out our thread id */
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	unsigned int tid = idx + idy * blockDim.x * gridDim.x;
	
	// Chj: Let the first thread clear the d_bin_data_shared[] array.
	if(threadIdx.x==0)
	{
		for(int i=0; i<BIN256; i++)
			d_bin_data_shared[i] = 0;
	}

	// Chj: All threads should wait for the first-thread's clearing done.
	__syncthreads();

	/* Fetch the data value as 32 bit */
	const unsigned int value_u32 = d_hist_data[tid];
	
	// Partial counting into d_bin_data_shared[]
	//
	if(tid < sample_ints)
	{
		atomicAdd( &(d_bin_data_shared[ (value_u32 & 0x000000FF) ]), 1 );
		atomicAdd( &(d_bin_data_shared[ (value_u32 & 0x0000FF00) >>  8 ]), 1 );
		atomicAdd( &(d_bin_data_shared[ (value_u32 & 0x00FF0000) >> 16 ]), 1 );
		atomicAdd( &(d_bin_data_shared[ (value_u32 & 0xFF000000) >> 24 ]), 1 );
	}
	
	/* Wait for all threads to update shared memory, again */
	__syncthreads();

	// Chj: Let the first thread accumulate the counting result.
	if(threadIdx.x==0)
	{
		for(int i=0; i<BIN256; i++)
		{
			atomicAdd( &d_bin_data[i], d_bin_data_shared[i] );
		}
	}
}

//////////////////////////////////////////////////////////////////////

void generate_histogram(const char *title, int sample_count, int threads_per_block)
{
	int i;
	Uchar *caSamples = new Uchar[sample_count]; // cpu mem
	Uchar *kaSamples = nullptr; // gpu mem
	Uint caCount_init[BIN256] = {}; // histogram init counted, the correct answer
	Uint caCount[BIN256] = {}; 
	Uint *kaCount = nullptr;      // histogram counted by gpu
	
	cudaEvent_t start = nullptr, stop = nullptr; // for GPU timing
	HANDLE_ERROR( cudaEventCreate(&start) );
	HANDLE_ERROR( cudaEventCreate(&stop) );

	// fill caSamples[] and caCount_init[]
	//
	for(i=0; i<sample_count; i++)
	{
		int ball = rand() % BIN256;
		caSamples[i] = ball;
		caCount_init[ball]++ ;
	}

	printf("[%s] Counting %d samples ...\n", title, sample_count);

	// Copy host-RAM to gpu-RAM

	HANDLE_ERROR( cudaMalloc((void**)&kaSamples, sample_count) );
	HANDLE_ERROR( cudaMemcpy(kaSamples, caSamples, sample_count, cudaMemcpyHostToDevice) );

	HANDLE_ERROR( cudaMalloc((void**)&kaCount, BIN256*sizeof(int)) );
	HANDLE_ERROR( cudaMemcpy(kaCount, caCount, BIN256*sizeof(int), cudaMemcpyHostToDevice) );

	// start kernel-call timing
	HANDLE_ERROR( cudaEventRecord( start, 0 ) ); 

	//
	// Select a kernel function to execute, according to `title`
	//

	if(strcmp(title, "p98:myhistogram_01")==0)
	{
		myhistogram_01<<<OCC_DIVIDE(sample_count, threads_per_block), threads_per_block>>>
			(kaSamples, kaCount, sample_count);
	}
	else if(strcmp(title, "p99:myhistogram_02")==0)
	{
		if(sample_count%4 != 0)
		{
			printf("ERROR user parameter input: For %s, sample_count must be multiple of 4. You passed in %d.\n",
				title, sample_count);
			exit(1);
		}

		int sample_ints = sample_count/4;
		myhistogram_02<<<OCC_DIVIDE(sample_ints, threads_per_block), threads_per_block>>>
			((Uint*)kaSamples, kaCount, sample_ints);
	}
	else if(strcmp(title, "p101:myhistogram_03a")==0)
	{
		if(sample_count%4 != 0)
		{
			printf("ERROR user parameter input: For %s, sample_count must be multiple of 4. You passed in %d.\n",
				title, sample_count);
			exit(1);
		}

		int sample_ints = sample_count/4;
		myhistogram_03a<<<OCC_DIVIDE(sample_ints, threads_per_block), threads_per_block>>>
			((Uint*)kaSamples, kaCount, sample_ints);
	}
	else
	{
		printf("ERROR: Unknown title requested: %s\n", title);
		exit(1);
	}

	// Check kernel launch success/fail.

	cudaError_t kerr = cudaPeekAtLastError();
	if(kerr) {
		printf("[%s] ERROR launching kernel call, errcode: %d (%s)\n", title, 
			kerr, cudaGetErrorString(kerr));
		exit(4);
	}

	// stop kernel-call timing
	HANDLE_ERROR( cudaEventRecord( stop, 0 ) ); 
	HANDLE_ERROR( cudaEventSynchronize( stop ) );

	// Copy gpu-RAM to host-RAM (acquire result)

	HANDLE_ERROR( cudaMemcpy(caCount, kaCount, BIN256*sizeof(int), cudaMemcpyDeviceToHost) );

	// Verify GPU-counted result.
	//
	printf("Verifying... ");
	for(i=0; i<BIN256; i++)
	{
		if(caCount[i]!=caCount_init[i])
		{
			printf("ERROR at sample index %d, correct: %d , wrong: %d\n",
				i, caCount_init[i], caCount[i]);
			exit(4);
		}
	}

	float elapse_millisec = 0;
	HANDLE_ERROR( cudaEventElapsedTime( &elapse_millisec, start, stop ) );

	if(elapse_millisec==0)
	{
		printf("Success. (0 millisec)\n");
	}
	else
	{
		printf("Success. (%.5g millisec, %.5g GB/s)\n", 
			elapse_millisec, 
			((double)sample_count/(1000*1000))/elapse_millisec);
	}

	// Release resources.
	HANDLE_ERROR( cudaFree(kaCount) );
	HANDLE_ERROR( cudaFree(kaSamples) );
	delete caSamples;
}


extern"C" void 
main_myhistogram(int argc, char* argv[])
{
	if(argc==1)
	{
		printf("Usage:\n");
		printf("    myhistogram <sample_count> [threads_per_block]\n");
		printf("\n");
		printf("Examples:\n");
		printf("    myhistogram 1024\n");
		printf("    myhistogram 1024000 512\n");
		exit(1);
	}

	int sample_count = strtoul(argv[1], nullptr, 0);

	int threads_per_block = 256;

	if(argc>2) {
		threads_per_block = strtoul(argv[2], nullptr, 0);
	}
	
	if(sample_count<=0) {
		printf("Wrong sample_count number(must >0): %d\n", sample_count);
		exit(1);
	}

	if(threads_per_block<=0) {
		printf("Wrong threads_per_block number(must >0): %d\n", threads_per_block);
	}

	generate_histogram("p98:myhistogram_01", sample_count, threads_per_block);
	printf("\n");
	generate_histogram("p99:myhistogram_02", sample_count, threads_per_block);
	printf("\n");
	generate_histogram("p101:myhistogram_03a", sample_count, threads_per_block);
}
