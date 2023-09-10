#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_anim.h"
#include "../common/chjdbg.h"

#define DIM 1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f

#define SPEED   0.25f  
// -- Chj: this value should not exceed 1/4, bcz we calculate new temperature 
//    of the center pixel according to 4 adjacent pixels, otherwise, the "next"
//    temperature of the center pixel will exceed one of the surrounding pixels, 
//    which is physically impossible.

// these exist on the GPU side
texture<float>  texConstSrc;
texture<float>  texIn;
texture<float>  texOut;

// this kernel takes in a 2-d array of floats
// it updates the value-of-interest by a scaled value based
// on itself and its nearest neighbors
__global__ void blend_kernel( float *dst, bool dstOut ) 
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	int left = offset - 1;
	int right = offset + 1;
	if (x == 0)   left++;
	if (x == DIM-1) right--; 

	int top = offset - DIM;
	int bottom = offset + DIM;
	if (y == 0)   top += DIM;
	if (y == DIM-1) bottom -= DIM;

	float   t, l, c, r, b;
	if (dstOut) {
		t = tex1Dfetch(texIn,top);
		l = tex1Dfetch(texIn,left);
		c = tex1Dfetch(texIn,offset);
		r = tex1Dfetch(texIn,right);
		b = tex1Dfetch(texIn,bottom);

	} else {
		t = tex1Dfetch(texOut,top);
		l = tex1Dfetch(texOut,left);
		c = tex1Dfetch(texOut,offset);
		r = tex1Dfetch(texOut,right);
		b = tex1Dfetch(texOut,bottom);
	}
	dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}

// NOTE - texOffsetConstSrc could either be passed as a
// parameter to this function, or passed in __constant__ memory
// if we declared it as a global above, it would be
// a parameter here: 
// __global__ void copy_const_kernel( float *iptr,
//                                    size_t texOffset )
__global__ void copy_const_kernel( float *iptr ) 
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float c = tex1Dfetch(texConstSrc, offset);
	if (c != 0)
		iptr[offset] = c;
}

// globals needed by the update routine
struct DataBlock {
	unsigned char   *output_bitmap;
	float           *dev_inSrc;
	float           *dev_outSrc;
	float           *dev_constSrc;
	CPUAnimBitmap  *bitmap;

	cudaEvent_t     start, stop;
	float           totalTime;
	unsigned        frames; // Chj: use unsigned instead of float
};

void anim_gpu( DataBlock *d, int ticks ) 
{
	(void)ticks;
	HANDLE_ERROR( cudaEventRecord( d->start, 0 ) );
	dim3    blocks(DIM/16,DIM/16);
	dim3    threads(16,16);
	CPUAnimBitmap  *bitmap = d->bitmap;

	// since tex is global and bound, we have to use a flag to
	// select which is in/out per iteration
	volatile bool dstOut = true;
	for (int i=0; i<90; i++) 
	{
		float   *in, *out;
		if (dstOut) {
			in  = d->dev_inSrc;
			out = d->dev_outSrc;
		} else {
			out = d->dev_inSrc;
			in  = d->dev_outSrc;
		}
		
		copy_const_kernel<<<blocks,threads>>>( in );
		
		blend_kernel<<<blocks,threads>>>( out, dstOut );
		
		dstOut = !dstOut;
	}
	
	float_to_color<<<blocks,threads>>>( d->output_bitmap, d->dev_inSrc );
	// -- after 90(an even number) iterations, dev_inSrc contains the final result.

	HANDLE_ERROR( cudaMemcpy( bitmap->get_ptr(),
							d->output_bitmap,
							bitmap->image_size(),
							cudaMemcpyDeviceToHost ) );

	HANDLE_ERROR( cudaEventRecord( d->stop, 0 ) );
	HANDLE_ERROR( cudaEventSynchronize( d->stop ) );

	float   elapsedTime;
	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, d->start, d->stop ) );
	d->totalTime += elapsedTime;
	++d->frames;
	
	char title[80] = {};
	C_SNPRINTF(title, sizeof(title)-1, 
		"[#%u] Average Time per frame:  %3.1f ms", 
		d->frames, d->totalTime/d->frames );

	printf("%s (ticks=%d)\n", title, ticks);
	glutSetWindowTitle(title);
}

// clean up memory allocated on the GPU
void anim_exit( DataBlock *d ) 
{
	cudaUnbindTexture( texIn );
	cudaUnbindTexture( texOut );
	cudaUnbindTexture( texConstSrc );
	HANDLE_ERROR( cudaFree( d->dev_inSrc ) );
	HANDLE_ERROR( cudaFree( d->dev_outSrc ) );
	HANDLE_ERROR( cudaFree( d->dev_constSrc ) );

	HANDLE_ERROR( cudaEventDestroy( d->start ) );
	HANDLE_ERROR( cudaEventDestroy( d->stop ) );
}


int main( void ) 
{
	DataBlock   data;
	CPUAnimBitmap bitmap( DIM, DIM, &data );
	data.bitmap = &bitmap;
	data.totalTime = 0;
	data.frames = 0;
	HANDLE_ERROR( cudaEventCreate( &data.start ) );
	HANDLE_ERROR( cudaEventCreate( &data.stop ) );

	int imageSize = bitmap.image_size();

	HANDLE_ERROR( cudaMalloc( (void**)&data.output_bitmap, imageSize ) );

	// assume float == 4 chars in size (ie rgba)
	HANDLE_ERROR( cudaMalloc( (void**)&data.dev_inSrc,    imageSize ) );
	HANDLE_ERROR( cudaMalloc( (void**)&data.dev_outSrc,   imageSize ) );
	HANDLE_ERROR( cudaMalloc( (void**)&data.dev_constSrc, imageSize ) );

	HANDLE_ERROR( cudaBindTexture( NULL, texConstSrc,
								data.dev_constSrc,
								imageSize ) );

	HANDLE_ERROR( cudaBindTexture( NULL, texIn,
								data.dev_inSrc,
								imageSize ) );

	HANDLE_ERROR( cudaBindTexture( NULL, texOut,
								data.dev_outSrc,
								imageSize ) );

	// initialize the constant data
	//
	float *temp = (float*)malloc( imageSize );
	for (int i=0; i<DIM*DIM; i++) {
		temp[i] = 0;
		int x = i % DIM;
		int y = i / DIM;
		if ((x>300) && (x<600) && (y>310) && (y<601))
			temp[i] = MAX_TEMP;
	}
	temp[100 + DIM*100] = (MAX_TEMP + MIN_TEMP)/2;
	temp[100 + DIM*700] = MIN_TEMP;
	temp[300 + DIM*300] = MIN_TEMP;
	temp[700 + DIM*200] = MIN_TEMP;
	for (int y=800; y<900; y++) {
		for (int x=400; x<500; x++) {
			temp[x+y*DIM] = MIN_TEMP;
		}
	}
	HANDLE_ERROR( cudaMemcpy( data.dev_constSrc, temp,
							imageSize,
							cudaMemcpyHostToDevice ) );    

	// initialize the input data
	//
	for (int y=800; y<DIM; y++) {
		for (int x=0; x<200; x++) {
			temp[x+y*DIM] = MAX_TEMP;
		}
	}
	HANDLE_ERROR( cudaMemcpy( data.dev_inSrc, temp,
							imageSize,
							cudaMemcpyHostToDevice ) );
	free( temp );

	bitmap.anim_and_exit( (void (*)(void*,int))anim_gpu,
		(void (*)(void*))anim_exit );
}
