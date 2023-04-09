#include "i22-widget.h"
#include <string>
#include <vector>

namespace item22
{
	struct Widget::Impl
	{
		std::string name;
		std::vector<double> data;
	};

	Widget::Widget()
		: pImpl( std::make_unique<Impl>() )
	{		
	}
}
