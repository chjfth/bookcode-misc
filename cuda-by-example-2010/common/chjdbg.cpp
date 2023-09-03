#include <stdio.h>
#include "chjdbg.h"

void dump_microseconds_diffs(unsigned int ar_uints[], int arsize)
{
	const int width1 = 10;
	printf("[  0] %*u microseconds\n", width1, ar_uints[0]);

	int i;
	for(i=1; i<arsize; i++)
	{
		printf("[%3d] %*u (+%u)\n", i, 
			width1, ar_uints[i], 
			(ar_uints[i]-ar_uints[i-1]));
	}
}
