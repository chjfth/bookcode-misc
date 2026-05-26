#include <sdring.h>
#include <TScalableArray.h>

#include "widget_ensureclnup.h"

#ifdef USE_ALTERNATIVE

Widget::~Widget() = default;
// -- [2023-04-10] Chj: Verified with VC2019 and gcc-12,
//    We can really put this dtor definition *before* struct Widget::Impl{...} .

#endif

struct Widget::Impl
{
	sdring<char> name;
	TScalableArray<int> data;

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
	: pImpl(new Impl(*rhs.pImpl))
{
}
Widget& Widget::operator=(const Widget & rhs)
{
	*pImpl = *rhs.pImpl;
	return *this;
}