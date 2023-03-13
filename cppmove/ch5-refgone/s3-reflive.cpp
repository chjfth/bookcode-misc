#include <stdio.h>

class CCount
{
public:
	int mi;
	int mis[3]; // make CCount object 16 bytes
	const int &getIntRef() const
	{ 
		return this->mi; 
	}
};	

CCount return_a_CCount(int n)
{
	CCount cobj {n+2};
	printf("cobj addr is %p\n", &cobj);
	return cobj;
}

void test_reflive()
{
	// This code seemingly runs OK with VC2019 & gcc 12,
	// but actually, its usage of r2a and r2b is conceptually wrong!
	
	const int &r2a = return_a_CCount(600).getIntRef();
	printf("[step1] &r2a=%p , r2a=%d\n", &r2a, r2a);

	printf("\n");

	const int &r2b = return_a_CCount(700).getIntRef();
	printf("[step2] &r2a=%p , r2a=%d\n", &r2a, r2a);
	printf("[step2] &r2b=%p , r2b=%d\n", &r2b, r2b);
}

int main()
{
	test_reflive();
	return 0;
}
