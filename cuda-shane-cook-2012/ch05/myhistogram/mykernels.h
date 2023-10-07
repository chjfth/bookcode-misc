#ifndef __mykernels_h_
#define __mykernels_h_

#include "../../share/share.h"

extern"C"{

__global__ void myhistogram_01( // @page 98
	const unsigned char * d_hist_data,
	unsigned int * d_bin_data,
	int sample_count);


__global__ void myhistogram_02( // @page 99-100
	const unsigned int * d_hist_data, // note: each call will process 4 bytes(Uint)
	unsigned int * d_bin_data,
	int sample_ints);

__global__ void myhistogram_02N( // #02 with Nbatch param
	const Uint *d_hist_data, 
	Uint *d_bin_data,
	int sample_ints,
	Uint Nbatch);


__shared__ unsigned int d_bin_data_shared[BIN256];
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
