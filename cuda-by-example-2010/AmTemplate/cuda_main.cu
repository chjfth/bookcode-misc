#include <stdio.h>
#include <stdlib.h>

// Note: nvcc.exe will include necessary CUDA .h headers automatically,
// so we do not have to #include them manually.

extern"C" void test_call_cuda()
{
	int gpu_count = 0;
	cudaError_t err = cudaGetDeviceCount(&gpu_count);
	if(err)
	{
		printf("cudaGetDeviceCount() error = %d\n", err);
		return;
	}
	else
	{
		printf("gpu_count = %d\n", gpu_count);
	}
}

int main(int argc, char* argv[])
{
	printf("Hello, cuda_main!\n");

	test_call_cuda();

	return 0;
}
