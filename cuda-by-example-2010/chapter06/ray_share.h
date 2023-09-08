#ifndef __rayshare_h_
#define __rayshare_h_

#include "cuda.h"
#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1024

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

#define SPHERES_MAX 1000

int SPHERES = 20;

struct Sphere 
{
	float   r,b,g;
	float   radius;
	float   x,y,z;
	__device__ float hit( float ox, float oy, float *n ) 
	{
		// Chj: (ox,oy) is the coordinate on the imaging surface(eye-field).

		float dx = ox - x;
		float dy = oy - y;

		// Hint: dx*dx + dy*dy + dz*dz = raduis*radius
		// Now we will calculate dz from the other 3 info.

		if (dx*dx + dy*dy < radius*radius) {
			float dz = sqrtf( radius*radius - dx*dx - dy*dy );
			*n = dz / sqrtf( radius * radius );
			
			// Now, (ox,oy,z+dz) is the point of the sphere surface.
			return dz + z;
		}
		return -INF;
	}
};

#ifdef USE_GPU_CONSTANT_MEM
__constant__ Sphere s[SPHERES_MAX]; // __constant__
#else 
Sphere *s = nullptr; // s will point to cudaMalloc-ed space.
#endif

__global__ void kernel( 
#ifndef USE_GPU_CONSTANT_MEM
	Sphere *s,
#endif
	int spheres, unsigned char *ptr) 
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float   ox = (x - DIM/2);
	float   oy = (y - DIM/2);

	/*
	考察每个球 Ball_n 投影到我们的视野中, 在视野中(ox,oy)这个坐标点的情况, 
	找出是哪个球在 (ox,oy) 处的球体最靠近我们的人眼, 把这个球挑出来,
	最靠近人眼的这个点, 称其 Z 坐标为 oz, 接着, 计算这个球在 (ox,oy,oz)
	处呈现的 RGB 颜色值. 将该 RGB 值填入 offset[] 的恰当位置, 任务完成.
	*/
	float   r=0, g=0, b=0;
	float   maxz = -INF;
	for(int i=0; i<spheres; i++) {
		float   n;
		float   t = s[i].hit( ox, oy, &n );
		if (t > maxz) {
			float fscale = n; // n(0.0 ~ 1.0) is used to calculate RGB value
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


static bool init_params(int argc, char *argv[])
{
	if(argc==1) {
		printf("Hint: You can pass two params: <spheres> <seed>\n");
		printf("For example: \n");
		printf("    ray 10 222\n");
		printf("\n");
	}

	if(argc>1) {
		SPHERES = (int)strtoul(argv[1], nullptr, 0);
	}

	if(SPHERES>SPHERES_MAX) {
		printf("Sorry, spheres cannot exceed %d.\n", SPHERES_MAX);
		return false;
	}
	printf("Spheres: %d\n", SPHERES);

	if(argc>2) {
		unsigned seed = strtoul(argv[2], nullptr, 0);
		printf("Calling srand(%u).\n", seed);
		srand(seed);
	}
	else {
		printf("Not calling srand().\n");
	}

	return true;
}

#endif