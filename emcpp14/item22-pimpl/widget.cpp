#include <string>
#include <vector>

#include "widget.h"

#ifdef USE_ALTERNATIVE

Widget::~Widget() = default;
// -- [2023-04-10] Chj: Verified with VC2019 and gcc-12,
//    We can really put this dtor definition *before* struct Widget::Impl{...} .

#endif

struct Widget::Impl
{
	std::string name;
	std::vector<double> data;
};

Widget::Widget()
	: pImpl(std::make_unique<Impl>())
{
}

#ifndef USE_ALTERNATIVE
// Define Widget's dtor after Widget::Impl's class body has been seen.
Widget::~Widget()
{	
}
#endif

// move-ctor/assign
//
Widget::Widget(Widget && rhs) = default;
Widget& Widget::operator=(Widget && rhs) = default;

// copy-ctor/assign
//
Widget::Widget(const Widget & rhs) // copy ctor
	: pImpl(std::make_unique<Impl>(*rhs.pImpl))
{

}
Widget& Widget::operator=(const Widget & rhs)
{
	*pImpl = *rhs.pImpl;
	return *this;
}