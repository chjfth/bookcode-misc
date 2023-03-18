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

	CCount(const CCount &cnt) 
	{
		this->mi = cnt.mi;
		printf("CCount ctor([CCount:\"%d\"]).   %p+\n", 
			cnt.mi, this);
	}
	CCount(int ii) 
	{
		this->mi = ii;
		printf("CCount ctor(\"%d\").            %p+\n", 
			ii, this);
	}
	~CCount()
	{
		printf("CCount dtor().                 %p-\n", this);
	}
};	

CCount return_a_CCount(int n)
{
	CCount cobj {n+2};
	printf("                  cobj addr is %p\n", &cobj);
	return cobj;
}

void test_reflong()
{
	const CCount &rc1 = return_a_CCount(600);
	const CCount &rc2 = return_a_CCount(700);

	printf("&rc1=%p , rc1.mi=%d\n", &rc1, rc1.mi);
	printf("&rc2=%p , rc2.mi=%d\n", &rc2, rc2.mi);
}

int main()
{
	test_reflong();
	return 0;
}
