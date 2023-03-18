// rrlv1.cpp
#include <utility>
class CInt {
public: int mi;
}; 
void rrlv(CInt&& iobj) {
	int old = iobj.mi;
	iobj.mi = 4;
}

int main() {
	CInt i1{1};
	rrlv(i1); // ERROR
	return 0;
}
