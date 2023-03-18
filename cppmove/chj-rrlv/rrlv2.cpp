// rrlv2.cpp
#include <utility>
class CInt {
public: int mi;
}; 

CInt&& rrlv2(CInt&& iobj) {
	int old = iobj.mi;
	iobj.mi = 4;
	return iobj; // ERROR
}

int main() {
	CInt i1{1};
	rrlv2(std::move(i1));
	return 0;
}
