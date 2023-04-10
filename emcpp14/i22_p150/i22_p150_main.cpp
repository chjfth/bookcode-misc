#include "widget.h"

int main()
{
	// This is deliberate error code.
	// 
	// According to EMCPP14 p150 :
	// This file compiles error, bcz Widget::Impl's class definition is missing.
	// In the case of using unique_ptr, only a class declaration(incomplete type)
	// is NOT enough.
	
	Widget w1; // compile error

	// VC2019 16.11:
	//    error C2027: use of undefined type 'Widget::Impl'
	//
	// gcc-12:
	//    error: invalid application of ‘sizeof’ to incomplete type ‘Widget::Impl’
}

