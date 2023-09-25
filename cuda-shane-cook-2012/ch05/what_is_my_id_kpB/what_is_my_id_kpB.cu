#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

/* [2023-09-25] Chj: Difference to what_is_my_id_kp.cu :

* Use if/else branch in the kernel function.
* Do not print warp# anymore, bcz this value is wrong when blockDim.x is not multiple of 32.

Setting CUDA breakpoints inside those branches helps us understand CUDA debugger
UI representation, which is very powerful.

*/

__global__ void what_is_my_id_kpB()
{
	/* Thread id is [block index * block size + thread offset into the block] */
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(thread_idx%2 == 0)
	{
		printf("[B0]kp-Thread: %3u - Block: %2u - Thread %3u\n",
			thread_idx, 
			blockIdx.x, 
			threadIdx.x);
	}
	else
	{
		printf("[B1]kp-Thread: %3u - Block: %2u - Thread %3u\n",
			thread_idx, 
			blockIdx.x, 
			threadIdx.x);
	}
}

//////////////////////////////////////////////////////////////////////

void do_main(int argc, char* argv[])
{
	if(argc<3) {
		printf("Usage: \n");
		printf("     what_is_my_id_kpB <num_blocks> <num_threads>\n");
		printf("Example: \n");
		printf("     what_is_my_id_kpB 3 64\n");
		exit(4);
	}

	int num_blocks  = strtoul(argv[1], nullptr, 0);
	int num_threads = strtoul(argv[2], nullptr, 0);

	/* Execute our kernel */
	what_is_my_id_kpB<<<num_blocks, num_threads>>>();	
}

extern"C" void 
main_what_is_my_id_kpB(int argc, char* argv[])
{
	do_main(argc, argv);
}
