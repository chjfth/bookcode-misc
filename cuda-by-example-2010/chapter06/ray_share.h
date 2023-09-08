#ifndef __rayshare_h_
#define __rayshare_h_

#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1024

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

#define SPHERES 20

struct Sphere 
{
	float   r,b,g;
	float   radius;
	float   x,y,z;
	__device__ float hit( float ox, float oy, float *n ) 
	{
		float dx = ox - x;
		float dy = oy - y;
		if (dx*dx + dy*dy < radius*radius) {
			float dz = sqrtf( radius*radius - dx*dx - dy*dy );
			*n = dz / sqrtf( radius * radius );
			return dz + z;
		}
		return -INF;
	}
};

#ifdef USE_GPU_CONSTANT_MEM
__constant__ Sphere s[SPHERES];
#endif

__global__ void kernel( 
#ifndef USE_GPU_CONSTANT_MEM
	Sphere *s,
#endif
	unsigned char *ptr) 
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float   ox = (x - DIM/2);
	float   oy = (y - DIM/2);

	float   r=0, g=0, b=0;
	float   maxz = -INF;
	for(int i=0; i<SPHERES; i++) {
		float   n;
		float   t = s[i].hit( ox, oy, &n );
		if (t > maxz) {
			float fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
			maxz = t;
		}
	} 

	ptr[offset*4 + 0] = (int)(r * 255);
	ptr[offset*4 + 1] = (int)(g * 255);
	ptr[offset*4 + 2] = (int)(b * 255);
	ptr[offset*4 + 3] = 255;
}

#endif