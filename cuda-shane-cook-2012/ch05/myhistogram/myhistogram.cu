#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "../../share/share.h"

#define BIN256 256
#define THREADS256 256

__global__ void myhistogram_01(
	const unsigned char * d_hist_data,
	unsigned int * d_bin_data) // @page 98
{
	/* Work out our thread id */
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int tid = idx + idy * blockDim.x * gridDim.x;

	/* Fetch the data value */
	const unsigned char value = d_hist_data[tid];
	
//	printf("[#%d] .%u\n", tid, value);

	atomicAdd(&(d_bin_data[value]), 1);
}

//////////////////////////////////////////////////////////////////////

void generate_histogram(const char *title, int sample_count)
{
	int i;
	Uchar *caSamples = new Uchar[sample_count]; // cpu mem
	Uchar *kaSamples = nullptr; // gpu mem
	Uint caCount_init[BIN256] = {}; // histogram init counted, the correct answer
	Uint caCount[BIN256] = {}; 
	Uint *kaCount = nullptr;      // histogram counted by gpu

	// fill caSamples[] and caCount_init[]
	//
	for(i=0; i<sample_count; i++)
	{
		int ball = rand() % BIN256;
		caSamples[i] = ball;
		caCount_init[ball]++ ;
	}

	printf("Counting %d samples ...\n", sample_count);

	// Copy host-RAM to gpu-RAM

	HANDLE_ERROR( cudaMalloc((void**)&kaSamples, sample_count) );
	HANDLE_ERROR( cudaMemcpy(kaSamples, caSamples, sample_count, cudaMemcpyHostToDevice) );

	HANDLE_ERROR( cudaMalloc((void**)&kaCount, BIN256*sizeof(int)) );
	HANDLE_ERROR( cudaMemcpy(kaCount, caCount, BIN256*sizeof(int), cudaMemcpyHostToDevice) );

	// Execute our kernel 
	myhistogram_01<<<OCC_DIVIDE(sample_count, THREADS256), THREADS256>>>
		(kaSamples, kaCount);

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
	printf("Success.\n");

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
		printf("    myhistogram <sample_count>\n");
		printf("\n");
		printf("Examples:\n");
		printf("    myhistogram 1024\n");
		exit(1);
	}

	int sample_count = strtoul(argv[1], nullptr, 0);

	generate_histogram("p98:myhistogram_01", sample_count);
}
