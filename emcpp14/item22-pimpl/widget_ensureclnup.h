#include <EnsureClnup.h>

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
	CleanupDelega<Impl> pImpl;
};
