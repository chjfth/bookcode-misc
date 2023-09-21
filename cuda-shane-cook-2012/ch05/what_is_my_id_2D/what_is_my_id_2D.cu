/*  Orignial code from [CUDA2012] Shane Cook's book, CH05, page 87.
2023.09.21, Chj applies many fixes to his original code. Now it works well.
*/

#include <stdio.h>
#include <stdlib.h>

__global__ void what_is_my_id_2d_A(
	unsigned int * const block_x,
	unsigned int * const block_y,
	unsigned int * const thread_dot_x,
	unsigned int * const thread_dot_y,
	unsigned int * const calc_thread,
	unsigned int * const x_thread,
	unsigned int * const y_thread,
	unsigned int * const grid_dimx,
	unsigned int * const block_dimx,
	unsigned int * const grid_dimy,
	unsigned int * const block_dimy)
{
	const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
	const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx;

	block_x[thread_idx] = blockIdx.x;
	block_y[thread_idx] = blockIdx.y;
	thread_dot_x[thread_idx] = threadIdx.x;
	thread_dot_y[thread_idx] = threadIdx.y;
	calc_thread[thread_idx] = thread_idx;
	x_thread[thread_idx] = idx;
	y_thread[thread_idx] = idy;
	grid_dimx[thread_idx] = gridDim.x;
	block_dimx[thread_idx] = blockDim.x;
	grid_dimy[thread_idx] = gridDim.y;
	block_dimy[thread_idx] = blockDim.y;
}

////////////////////////////////////////////////////////////////

#define ARRAY_SIZE_X 32
#define ARRAY_SIZE_Y 16
#define ARRAY_SIZE_IN_BYTES ((ARRAY_SIZE_X) * (ARRAY_SIZE_Y) * (sizeof(unsigned int)))

/* Declare statically six arrays of ARRAY_SIZE each */
unsigned int cpu_block_x[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_block_y[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_thread_dot_x[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_thread_dot_y[ARRAY_SIZE_Y][ARRAY_SIZE_X];
//unsigned int cpu_warp[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_calc_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_Xthread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_Ythread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_grid_dimx[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_block_dimx[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_grid_dimy[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_block_dimy[ARRAY_SIZE_Y][ARRAY_SIZE_X];

void do_main(void)
{
	/* Total thread count = 32 * 4 = 128 */
	const dim3 threads_stripe(32, 4); // 32 * 4
	const dim3 blocks_stripe(1,4);

	/* Total thread count = 16 * 8 = 128 */
	const dim3 threads_square(16, 8); // 16 * 8 
	const dim3 blocks_square(2, 2);
	
	/* Declare pointers for GPU based params */
	unsigned int * gpu_block_x;
	unsigned int * gpu_block_y;
	unsigned int * gpu_thread_dot_x;
	unsigned int * gpu_thread_dot_y;
//	unsigned int * gpu_warp;   // no use
	unsigned int * gpu_calc_thread;
	unsigned int * gpu_Xthread;
	unsigned int * gpu_Ythread;
	unsigned int * gpu_grid_dimx;
	unsigned int * gpu_block_dimx;
	unsigned int * gpu_grid_dimy;
	unsigned int * gpu_block_dimy;

	/* Allocate four arrays on the GPU */
	cudaMalloc((void **)&gpu_block_x, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_block_y, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_thread_dot_x, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_thread_dot_y, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_calc_thread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_Xthread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_Ythread, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_grid_dimx, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_block_dimx, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_grid_dimy, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void **)&gpu_block_dimy, ARRAY_SIZE_IN_BYTES);
	
	for (int ikernel=0; ikernel<2; ikernel++)
	{
		const dim3 &blocks = (ikernel==0 ? blocks_stripe : blocks_square);
		const dim3 &threads = (ikernel==0 ? threads_stripe : threads_square);

		/* Execute our kernel */
		what_is_my_id_2d_A<<<blocks, threads>>>(
			gpu_block_x, 
			gpu_block_y,
			gpu_thread_dot_x, 
			gpu_thread_dot_y, 
			gpu_calc_thread, 
			gpu_Xthread, 
			gpu_Ythread, 
			gpu_grid_dimx, 
			gpu_block_dimx,
			gpu_grid_dimy, 
			gpu_block_dimy);

		/* Copy back the GPU results to the CPU */
		cudaMemcpy(cpu_block_x, gpu_block_x, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_block_y, gpu_block_y, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_thread_dot_x, gpu_thread_dot_x, ARRAY_SIZE_IN_BYTES,	cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_thread_dot_y, gpu_thread_dot_y, ARRAY_SIZE_IN_BYTES,	cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_calc_thread, gpu_calc_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_Xthread, gpu_Xthread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_Ythread, gpu_Ythread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_grid_dimx, gpu_grid_dimx, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_block_dimx,gpu_block_dimx, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_grid_dimy, gpu_grid_dimy, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpu_block_dimy, gpu_block_dimy, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);

		printf("\n==== Kernel call #%d <<<(%d,%d), (%d,%d)>>>====\n", 
			ikernel,
			blocks.x, blocks.y,
			threads.x, threads.y
			);
		printf("\n");

		/* Iterate through the arrays and print */
		for (int y=0; y < ARRAY_SIZE_Y; y++)
		{
			for (int x=0; x < ARRAY_SIZE_X; x++)
			{
				printf(
					"[%2d,%2d] "
					"CT: %-3u "
					"bk.y: %1u bk.x: %1u , "
					"tid.y: %2u tid.x: %2u , "
					"Ytid: %2u Xtid: %2u , "

					"gridDim.[y/x]:%1u/%1u "
					
					"blockDim.[y/x]:%1u/%1u\n"
					, 
					x, y,
					cpu_calc_thread[y][x], 
					cpu_block_y[y][x], cpu_block_x[y][x], 
					cpu_thread_dot_y[y][x], cpu_thread_dot_x[y][x], 
					cpu_Xthread[y][x], cpu_Ythread[y][x], 					
					
					cpu_grid_dimx[y][x], cpu_grid_dimy[y][x],  
					
					cpu_block_dimx[y][x], cpu_block_dimy[y][x]
				);	
			}
		}

		// Chj: print an extra blank line.
		printf("\n");
	}

	/* Free the arrays on the GPU as now we’re done with them */
	cudaFree(gpu_block_x);
	cudaFree(gpu_block_y);
	cudaFree(gpu_thread_dot_x);
	cudaFree(gpu_thread_dot_y);
	cudaFree(gpu_calc_thread);
	cudaFree(gpu_Xthread);
	cudaFree(gpu_Ythread);
	cudaFree(gpu_grid_dimx);
	cudaFree(gpu_block_dimx);
	cudaFree(gpu_grid_dimy);
	cudaFree(gpu_block_dimy);
}


extern"C" void 
main_what_is_my_id_2D(int argc, char* argv[])
{
	(void)argc; (void)argv;
	do_main();
}
