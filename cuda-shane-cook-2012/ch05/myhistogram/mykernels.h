#ifndef __mykernels_h_
#define __mykernels_h_

extern"C"{

#include "../../share/share.h"

#define BIN256 256


__global__ void myhistogram_01( // @page 98
	const unsigned char * d_hist_data,
	unsigned int * d_bin_data,
	int sample_count);


__global__ void myhistogram_02( // @page 99-100
	const unsigned int * d_hist_data, // note: each call will process 4 bytes(Uint)
	unsigned int * d_bin_data,
	int sample_ints) ;


extern __shared__ unsigned int d_bin_data_shared[];
// -- GPU sharedmem used by #03a, #03b, #07

__global__ void myhistogram_03a( // @page 101 modified
	const unsigned int * d_hist_data,
	unsigned int * d_bin_data,
	int sample_ints);

__global__ void myhistogram_03b( // byte-by-byte operation based on myhistogram_03a
	const Uchar * d_hist_data,
	Uint * const d_bin_data,
	int sample_count);


__global__ void myhistogram_07( // @page 102
	const Uint * d_hist_data,
	Uint * d_bin_data,
	int sample_ints,
	Uint Nbatch);


}; // extern"C"

#endif
