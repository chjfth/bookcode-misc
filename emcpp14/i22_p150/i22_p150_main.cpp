#include "widget.h"

int main()
{
	// According to EMCPP14 p150 :
	// This file compiles error, bcz Widget::Impl's class definition is missing.
	// In the case of using unique_ptr, only a class declaration(incomplete type)
	// is NOT enough.
	
	Widget w1;
}
