#include <string>
#include <vector>

#include "widget_AutoClean.h"

struct Widget::Impl
{
	std::string name;
	std::vector<int> data;

	Impl()
	{
		printf("Impl ctor() @%p...\n", this);
	}

	Impl(const Impl& rhs)
	{
		printf("Impl copy-ctor() @%p...\n", this);
		this->name = rhs.name;
		this->data = rhs.data;
	}

	~Impl()
	{
		printf("Impl dtor() @%p...\n", this);
	}
};

Widget::Widget()
	: pImpl(new Impl)
{
	if (!pImpl)
		return; // throw std::bad_alloc();
}

// Define Widget's dtor after Widget::Impl's class body has been seen.
Widget::~Widget()
{	
}


// move-ctor/assign
//
Widget::Widget(Widget && rhs) = default;
Widget& Widget::operator=(Widget && rhs) = default;

// copy-ctor/assign
//
Widget::Widget(const Widget & rhs) // copy ctor
	: pImpl(new Impl(*rhs.pImpl))
{
}
Widget& Widget::operator=(const Widget & rhs)
{
	*pImpl = *rhs.pImpl;
	return *this;
}