#include <stdio.h>

const int & return_goneref(int n)
{
	char x[32] = "\x2@@@";
	int r = n + x[0];
	return r;
}

void test_refgone()
{
	const int& r1 = return_goneref(555);

	printf("The address of r1=%p\n", &r1);

	int r1c = r1;  // gcc: coredump on reading r1's value	
	printf("r1c=%d\n", r1c); 
}

int main()
{
	test_refgone();
	return 0;
}
