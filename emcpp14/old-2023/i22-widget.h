#include <memory> // for unique_ptr<>


namespace item22
{

	class Widget
	{
	public:
		Widget();
		~Widget();

		// move-ctor/assign
		Widget(Widget&& rhs); 
		Widget& operator=(Widget&& rhs);

		// copy-ctor/assign
		Widget(const Widget& rhs);
		Widget& operator=(const Widget& rhs);
		
	private:
		struct Impl;
		std::unique_ptr<Impl> pImpl;
	};

}