#include <memory> // for unique_ptr<>

class Widget
{
public:
	Widget();

private:
	struct Impl;
	std::unique_ptr<Impl> pImpl;
};
