#ifndef __share_h_
#define __share_h_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern"C"{

typedef unsigned char Uchar;
typedef unsigned int Uint;

#ifdef _MSC_VER
#define int64 __int64
#else
#define int64 long long
#endif

typedef unsigned int64 uint64;

#define BIN256 256


#define OCC_DIVIDE(n, x) ( ((n)+(x)-1) / (x) ) // occupation divide

#ifdef _MSC_VER
#define C_SNPRINTF _snprintf_s
#else // For Linux
#define C_SNPRINTF snprintf
#endif

uint64 ps_GetOsMicrosecs64(void);

typedef struct _GGT_HSimpleThread_st { } *GGT_HSimpleThread;

typedef void (*PROC_ggt_simple_thread)(void *param);
// Note: On x86 Windows, this defaults to __cdecl, not __stdcall .

GGT_HSimpleThread ggt_simple_thread_create(PROC_ggt_simple_thread proc, void *param, int stack_size=0);
// return non-NULL handle on thread creation success.

bool ggt_simple_thread_waitend(GGT_HSimpleThread h);


void prepare_samples(Uchar *arSamples, int sample_count, Uint arCount[BIN256]);
bool verify_bin_result(const Uint std[BIN256], const Uint chk[BIN256]);
void generate_histogram_cpu(const char *title, int sample_count);


//////////////////////////////////////////////////////////////////////////////

static cudaError_t PrintCudaError(cudaError_t err, const char *file, int line)
{
	if(err)
	{
		printf( "[CUDAERR:%d] %s in %s at line %d\n", 
			err,
			cudaGetErrorString( err ),
			file, line 
			);
	}
	return err;
}

#define PRINT_ERROR(err) (PrintCudaError(err, __FILE__, __LINE__))

static void HandleError( cudaError_t err,
	const char *file,
	int line ) 
{
	if (err != cudaSuccess) {
		PrintCudaError(err, file, line);
		exit( EXIT_FAILURE );
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
	printf( "Host memory failed in %s at line %d\n", \
	__FILE__, __LINE__ ); \
	exit( EXIT_FAILURE );}}


void myPrintGpuInfo();

}; // extern"C"

#endif
