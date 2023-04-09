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
}
