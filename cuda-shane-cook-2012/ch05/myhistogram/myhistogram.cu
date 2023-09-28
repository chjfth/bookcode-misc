/*  Orignial code from [CUDA2012] Shane Cook's book, CH05, page 97-103.
2023.09.27, Chj adds caller code to the kernel functions and applies many fixes 
to his original kernel code. Now it runs smoothly, and the program's behavior
quite appositely matches the author's words.
*/

#include <stdio.h>
#include <stdlib.h>

#include "../../share/share.h"
#include "mykernels.h"

__shared__ unsigned int d_bin_data_shared[BIN256];
// -- sharedmem used by #03a, #03b and #07


void ReportErrorIfNot4xSamples(const char *title, int sample_count)
{
	if(sample_count%4 != 0)
	{
		printf("ERROR user parameter input: For %s, sample_count must be multiple of 4. You passed in %d.\n",
			title, sample_count);
		exit(1);
	}
}

void generate_histogram_gpu(const char *title, int sample_count, int threads_per_block,
	int Nbatch)
{
	Uchar *caSamples = new Uchar[sample_count]; // cpu mem
	Uchar *kaSamples = nullptr; // gpu mem
	Uint caCount_init[BIN256] = {}; // histogram init counted, the correct answer
	Uint caCount[BIN256] = {}; 
	Uint *kaCount = nullptr;      // histogram counted by gpu
	
	cudaEvent_t start = nullptr, stop = nullptr; // for GPU timing
	HANDLE_ERROR( cudaEventCreate(&start) );
	HANDLE_ERROR( cudaEventCreate(&stop) );

	// fill caSamples[] and caCount_init[]
	prepare_samples(caSamples, sample_count, caCount_init);

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
		ReportErrorIfNot4xSamples(title, sample_count);

		int sample_ints = sample_count/4;
		myhistogram_02<<<OCC_DIVIDE(sample_ints, threads_per_block), threads_per_block>>>
			((Uint*)kaSamples, kaCount, sample_ints);
	}
	else if(strcmp(title, "p101:myhistogram_03a")==0)
	{
		ReportErrorIfNot4xSamples(title, sample_count);

		int sample_ints = sample_count/4;
		myhistogram_03a<<<OCC_DIVIDE(sample_ints, threads_per_block), threads_per_block>>>
			((Uint*)kaSamples, kaCount, sample_ints);
	}
	else if(strcmp(title, "myhistogram_03b")==0)
	{
		myhistogram_03b<<<OCC_DIVIDE(sample_count, threads_per_block), threads_per_block>>>
			(kaSamples, kaCount, sample_count);
	}
	else if(strcmp(title, "p102:myhistogram_07")==0)
	{
		ReportErrorIfNot4xSamples(title, sample_count);

		printf("Using Nbatch = %d\n", Nbatch);

		int sample_ints = sample_count/4;
		Uint granularity = threads_per_block * Nbatch;

		myhistogram_07<<<OCC_DIVIDE(sample_ints, granularity), threads_per_block>>>
			((Uint*)kaSamples, kaCount, sample_ints, Nbatch);
	}
	else
	{
		printf("ERROR: Unknown GPU title requested: %s\n", title);
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

	const char *errprefix = nullptr;

	// Verify GPU-counted result.
	//
	printf("Verifying... ");
	bool vsucc = verify_bin_result(caCount_init, caCount);
	if(!vsucc)
		errprefix = "Error!!!";

	float elapse_millisec = 0;
	HANDLE_ERROR( cudaEventElapsedTime( &elapse_millisec, start, stop ) );

	if(elapse_millisec==0)
	{
		printf("%s (0 millisec)\n", 
			errprefix ? errprefix : "Success.");
	}
	else
	{
		printf("%s (%.5g millisec, %.5g GB/s)\n", 
			errprefix ? errprefix : "Success.",
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
		printf("    myhistogram <histogram_sample_count> [threads_per_block] [Nbatch]\n");
		printf("\n");
		printf("Examples:\n");
		printf("    myhistogram 1024\n");
		printf("    myhistogram 1024000 512\n");
		printf("    myhistogram 8 1 2\n");
		printf("    myhistogram 10240000 512 8\n");
		exit(1);
	}

	int sample_count = strtoul(argv[1], nullptr, 0);
	int threads_per_block = 256;
	int Nbatch = 1;

	if(argc>2) {
		threads_per_block = strtoul(argv[2], nullptr, 0);
	}
	if(argc>3) {
		// This param is used only 
		Nbatch = strtoul(argv[3], nullptr, 0);
	}
	
	if(sample_count<=0) {
		printf("Wrong sample_count number(must >0): %d\n", sample_count);
		exit(1);
	}
	if(threads_per_block<=0) {
		printf("Wrong threads_per_block number(must >0): %d\n", threads_per_block);
		exit(1);
	}
	if(Nbatch<=0) {
		printf("Wrong Nbatch number(must >0): %d\n", threads_per_block);
		exit(1);
	}

	generate_histogram_cpu("CPU_one_thread", sample_count);
	printf("\n");
	generate_histogram_cpu("CPU_two_threads", sample_count);
	printf("\n");

	myPrintGpuInfo();

	generate_histogram_gpu("p98:myhistogram_01", sample_count, threads_per_block, Nbatch);
	printf("\n");
	generate_histogram_gpu("p99:myhistogram_02", sample_count, threads_per_block, Nbatch);
	printf("\n");
	generate_histogram_gpu("p101:myhistogram_03a", sample_count, threads_per_block, Nbatch);
	printf("\n");
	generate_histogram_gpu("myhistogram_03b", sample_count, threads_per_block, Nbatch);
	printf("\n");
	generate_histogram_gpu("p102:myhistogram_07", sample_count, threads_per_block, Nbatch);
}
