/*  Orignial code from [CUDA2012] Shane Cook's book, CH05, page 97-103.
2023.09.27, Chj adds caller code to the kernel functions and applies many fixes 
to his original kernel code. Now it runs smoothly, and the program's behavior
quite appositely matches the author's words.
*/

#include <stdio.h>
#include <stdlib.h>

#include "../../share/share.h"
#include "mykernels.h"

const char *g_version = "20231007.1";

void ReportErrorIfNot4xSamples(const char *title, int sample_count)
{
	if(sample_count%4 != 0)
	{
		printf("ERROR user parameter input: For %s, sample_count must be multiple of 4. You passed in %d.\n",
			title, sample_count);
		exit(1);
	}
}

bool generate_histogram_gpu(const char *title, int sample_count, int threads_per_block,
	int Nbatch)
{
	Uchar *caSamples = new Uchar[sample_count]; // cpu mem
	Cec_delete_Uchar _caSamples(caSamples);

	Uchar *kaSamples = nullptr; // gpu mem
	Uint caCount_init[BIN256] = {}; // histogram init counted, the correct answer
	Uint caCount[BIN256] = {}; 
	Uint *kaCount = nullptr;      // histogram counted by gpu

	// fill caSamples[] and caCount_init[]
	prepare_samples(caSamples, sample_count, caCount_init);

	cudaEvent_t start = nullptr, stop = nullptr; // for GPU timing
	HANDLE_ERROR( cudaEventCreate(&start) );
	Cec_cudaEventDestroy _start(start);
	HANDLE_ERROR( cudaEventCreate(&stop) );
	Cec_cudaEventDestroy _stop(stop);

	// Allocate GPU RAM.
	HANDLE_ERROR( cudaMalloc((void**)&kaSamples, sample_count) );
	Cec_cudaFree _kaSamples(kaSamples);
	HANDLE_ERROR( cudaMalloc((void**)&kaCount, BIN256*sizeof(int)) );
	Cec_cudaFree _kaCount(kaCount);

	int sample_ints = sample_count/4;
	int num_blocks = 0;

	if(strcmp(title, "p98:myhistogram_01")==0)
	{
		num_blocks = OCC_DIVIDE(sample_count, threads_per_block);
	}
	else if(strcmp(title, "p99:myhistogram_02")==0)
	{
		ReportErrorIfNot4xSamples(title, sample_count);

		num_blocks = OCC_DIVIDE(sample_ints, threads_per_block);
	}
	else if(strcmp(title, "myhistogram_03b")==0)
	{
		num_blocks = OCC_DIVIDE(sample_count, threads_per_block);
	}
	else if(strcmp(title, "p101:myhistogram_03a")==0)
	{
		ReportErrorIfNot4xSamples(title, sample_count);

		num_blocks = OCC_DIVIDE(sample_ints, threads_per_block);
	}
	else if(strcmp(title, "p102:myhistogram_07")==0)
	{
		ReportErrorIfNot4xSamples(title, sample_count);

		Uint granularity = threads_per_block * Nbatch;
		num_blocks = OCC_DIVIDE(sample_ints, granularity);
	}
	else
	{
		printf("[%s] Unknown GPU title requested.\n", title);
		return false;
	}
	assert(num_blocks>0);

	dim3 griddim = my_blocks_to_dim3(num_blocks);

	printf("[%s] Counting %d samples (%d blocks * %d threads) ...\n", title, 
		sample_count, num_blocks, threads_per_block);

	// Start REAL timing (will calculate real-world elapsed milliseconds)
	uint64 usec_start = ps_GetOsMicrosecs64();

	// Copy host-RAM to gpu-RAM

	HANDLE_ERROR( cudaMemcpy(kaSamples, caSamples, sample_count, cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(kaCount, caCount, BIN256*sizeof(int), cudaMemcpyHostToDevice) );

	//////////////////////////////
	// start kernel-call timing //
	//////////////////////////////
	HANDLE_ERROR( cudaEventRecord( start, 0 ) ); 

	//
	// Select a kernel function to execute, according to `title`
	//

	if(strcmp(title, "p98:myhistogram_01")==0)
	{
		myhistogram_01<<<griddim, threads_per_block>>>
			(kaSamples, kaCount, sample_count);
	}
	else if(strcmp(title, "p99:myhistogram_02")==0)
	{
		myhistogram_02<<<griddim, threads_per_block>>>
			((Uint*)kaSamples, kaCount, sample_ints);
	}
	else if(strcmp(title, "myhistogram_03b")==0)
	{
		myhistogram_03b<<<griddim, threads_per_block>>>
			(kaSamples, kaCount, sample_count);
	}
	else if(strcmp(title, "p101:myhistogram_03a")==0)
	{
		myhistogram_03a<<<griddim, threads_per_block>>>
			((Uint*)kaSamples, kaCount, sample_ints);
	}
	else if(strcmp(title, "p102:myhistogram_07")==0)
	{
		printf("Using Nbatch = %d\n", Nbatch);

		myhistogram_07<<<griddim, threads_per_block>>>
			((Uint*)kaSamples, kaCount, sample_ints, Nbatch);
	}
	else
	{
		printf("[%s] Unknown GPU title requested.\n", title);
		return false;
	}

	// Check kernel launch success/fail.

	cudaError_t kerr = cudaGetLastError();
	if(kerr) {
		printf("[%s] ERROR launching kernel call, errcode: %d (%s)\n", title, 
			kerr, cudaGetErrorString(kerr));
		return false;
	}

	/////////////////////////////
	// stop kernel-call timing //
	/////////////////////////////
	HANDLE_ERROR( cudaEventRecord( stop, 0 ) ); 
	HANDLE_ERROR( cudaEventSynchronize( stop ) );

	// Copy gpu-RAM to host-RAM (acquire result)

	HANDLE_ERROR( cudaMemcpy(caCount, kaCount, BIN256*sizeof(int), cudaMemcpyDeviceToHost) );

	// Stop REAL timing 
	uint64 usec_stop = ps_GetOsMicrosecs64();
	uint64 usec_used = usec_stop - usec_start;

	const char *errprefix = nullptr;

	// Verify GPU-counted result.
	//
	printf("Verifying... ");
	bool vsucc = verify_bin_result(caCount_init, caCount);
	if(!vsucc)
		errprefix = " ##### ERROR DETECTED. #####\n";

	float gpu_elapse_millisec = 0;
	HANDLE_ERROR( cudaEventElapsedTime( &gpu_elapse_millisec, start, stop ) );

	if(gpu_elapse_millisec==0)
	{
		printf("%s REAL: %.5g millisec (GPU-only: 0 millisec)\n", 
			errprefix ? errprefix : "Success.",
			(double)usec_used/1000);
	}
	else
	{
		printf("%s REAL: %.5g millisec (GPU-only: %.5g millisec, %.5g GB/s)\n", 
			errprefix ? errprefix : "Success.",
			(double)usec_used/1000,
			gpu_elapse_millisec, 
			((double)sample_count/(1000*1000))/gpu_elapse_millisec);
	}

	return true;
}


extern"C" void 
main_myhistogram(int argc, char* argv[])
{
	printf("myhistogram version %s\n", g_version);

	if(argc==1)
	{
		printf("Usage:\n");
		printf("    myhistogram <histogram_sample_count> [threads_per_block] [Nbatch]\n");
		printf("\n");
		printf("Examples:\n");
		printf("    myhistogram 1024\n");
		printf("    myhistogram 1024000 512\n");
		printf("    myhistogram 16 1 2\n");
		printf("    myhistogram 10240000 512 8\n");
		exit(1);
	}

	int sample_count = strtoul(argv[1], nullptr, 0);
	int threads_per_block = 256;
	int Nbatch = 8;

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
	generate_histogram_gpu("myhistogram_03b", sample_count, threads_per_block, Nbatch);
	printf("\n");
	generate_histogram_gpu("p101:myhistogram_03a", sample_count, threads_per_block, Nbatch);
	printf("\n");
	generate_histogram_gpu("p102:myhistogram_07", sample_count, threads_per_block, Nbatch);
}
