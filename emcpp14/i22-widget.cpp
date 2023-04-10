#include "i22-widget.h"
#include <string>
#include <vector>

namespace item22
{
	Widget::~Widget() = default;
	// -- [2023-04-10] Chj: Verified with VC2019 and gcc-12,
	//    We can really put this dtor definition *before* struct Widget::Impl{...} .
	
	struct Widget::Impl
	{
		std::string name;
		std::vector<double> data;
	};

	Widget::Widget()
		: pImpl( std::make_unique<Impl>() )
	{		
	}

//	Widget::~Widget() = default; // Chj: This has been moved above.

	// move-ctor/assign
	//
	Widget::Widget(Widget&& rhs) = default; // declarations
	Widget& Widget::operator=(Widget&& rhs) = default;

	// copy-ctor/assign
	//
	Widget::Widget(const Widget& rhs) // copy ctor
		: pImpl(std::make_unique<Impl>(*rhs.pImpl))
	{
		
	}
	Widget& Widget::operator=(const Widget& rhs)
	{
		*pImpl = *rhs.pImpl;
		return *this;
	}
}
