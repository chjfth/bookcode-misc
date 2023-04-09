#include <memory> // for unique_ptr<>


namespace item22
{

	class Widget
	{
	public:
		Widget();

	private:
		struct Impl;
		std::unique_ptr<Impl> pImpl;
	};

}