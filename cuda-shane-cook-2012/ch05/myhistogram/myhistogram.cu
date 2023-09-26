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

	HANDLE_ERROR( cudaEventRecord( start, 0 ) ); // start timing

	// Execute our kernel 
	myhistogram_01<<<OCC_DIVIDE(sample_count, threads_per_block), threads_per_block>>>
		(kaSamples, kaCount, sample_count);

	cudaError_t kerr = cudaPeekAtLastError();
	if(kerr) {
		printf("[%s] ERROR launching kernel call, errcode: %d (%s)\n", title, 
			kerr, cudaGetErrorString(kerr));
		exit(4);
	}

	HANDLE_ERROR( cudaEventRecord( stop, 0 ) ); // stop timing
	HANDLE_ERROR( cudaEventSynchronize( stop ) );

	// Copy gpu-RAM to host-RAM (acquire result)

	HANDLE_ERROR( cudaMemcpy(caCount, kaCount, BIN256*sizeof(int), cudaMemcpyDeviceToHost) );

	// Verify GPU-counted result.
	//
	printf("Verifying ... ");
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

	printf("Success. (%g millisec)\n", elapse_millisec);

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
}
