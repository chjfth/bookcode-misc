// rrlv0.cpp
#include <utility>

class CInt
{
public:
	int mi;
};

void rrlv(CInt&& iobj)
{
	int old = iobj.mi;
	iobj.mi = 4;
}

int main()
{
	CInt i1{1};
	CInt i2{2};
	
	rrlv(std::move(i1));
	
	rrlv(static_cast<CInt&&>(i2));
	
	return 0;
}