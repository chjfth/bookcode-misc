#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

/* [2023-09-22] Chj: Difference to original what_is_my_id.cu :

This kp version use printf() directly inside the kernel function. We'd like to 
see whether thread_idx always prints-out in order.

The result is, it may not be in order. 

For example, running Debug or Release exe on GTX 870M (driver version 376.54)

what_is_my_id_kp.exe  2 64     // in order
what_is_my_id_kp.exe  2 8      // in order
what_is_my_id_kp.exe  3 8      // out of order
what_is_my_id_kp.exe  3 4      // out of order

*/

__global__ void what_is_my_id_kp()
{
	/* Thread id is [block index * block size + thread offset into the block] */
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	printf("kp-Thread: %3u - Block: %2u - Warp %2u - Thread %3u\n",
		thread_idx, 
		blockIdx.x, 
		threadIdx.x / warpSize, 
		threadIdx.x);
}

//////////////////////////////////////////////////////////////////////

void do_main(int argc, char* argv[])
{
	if(argc<3) {
		printf("Usage: \n");
		printf("     what_is_my_id_kp <num_blocks> <num_threads>\n");
		printf("Example: \n");
		printf("     what_is_my_id_kp 3 64\n");
		exit(4);
	}

	int num_blocks  = strtoul(argv[1], nullptr, 0);
	int num_threads = strtoul(argv[2], nullptr, 0);

	/* Execute our kernel */
	what_is_my_id_kp<<<num_blocks, num_threads>>>();	
}

extern"C" void 
main_what_is_my_id_kp(int argc, char* argv[])
{
	do_main(argc, argv);
}
