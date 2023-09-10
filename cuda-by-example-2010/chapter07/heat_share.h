#ifndef __heat_share_h_
#define __heat_share_h_

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



#endif
