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
texture<float,2>  texConstSrc;
texture<float,2>  texIn;
texture<float,2>  texOut;

__global__ void blend_kernel( float *dst, bool dstOut ) 
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float   t, l, c, r, b;
	if (dstOut) {
		t = tex2D(texIn, x, y-1); // top
		l = tex2D(texIn, x-1, y); // left
		c = tex2D(texIn, x, y);   // self
		r = tex2D(texIn, x+1, y); // right
		b = tex2D(texIn, x, y+1); // bottom
	} else {
		t = tex2D(texOut, x, y-1); // top
		l = tex2D(texOut, x-1, y); // left
		c = tex2D(texOut, x, y);   // self
		r = tex2D(texOut, x+1, y); // right
		b = tex2D(texOut, x, y+1); // bottom
	}
	dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}

__global__ void copy_const_kernel( float *iptr ) 
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float c = tex2D(texConstSrc, x, y);
	if (c != 0)
		iptr[offset] = c;
}

#include "heat_share.h"

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

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

	HANDLE_ERROR( cudaBindTexture2D( NULL, texConstSrc,
									data.dev_constSrc,
									desc, DIM, DIM,
									sizeof(float) * DIM ) );

	HANDLE_ERROR( cudaBindTexture2D( NULL, texIn,
									data.dev_inSrc,
									desc, DIM, DIM,
									sizeof(float) * DIM ) );

	HANDLE_ERROR( cudaBindTexture2D( NULL, texOut,
									data.dev_outSrc,
									desc, DIM, DIM,
									sizeof(float) * DIM ) );

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
