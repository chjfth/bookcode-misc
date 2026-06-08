#include <iostream>
#include <string_view>

class PersonA
{
	std::string name;
public:
	PersonA (std::string_view _name) // don’t do this
		: name{ _name }
	{
		std::cout << "PersonA ctor(), _name=" << _name << '\n';
	}
};

class PersonB
{
	std::string name;
public:
	PersonB (std::string _name)
		: name{ std::move(_name) }
	{
		std::cout << "PersonB ctor(), _name=" << _name << '\n';
	}
};


int main(int argc, char *argv[])
{
	PersonA p1{ "Jim" };     // no performance overhead

	std::string s = "Joe";
	PersonA p2{ s };         // no performance overhead

	PersonA p3{ std::move(s) }; // performance overhead: broken move()
	// -- [2026-06-08] Chj: I find book text ambiguity today.
	//    p3's construction is NOT more expensive than p2.
	//    After p3 is constructed, `s` is NOT hollowed out.

	// What the book author really mean is:
	// PersonB's ctor is more effective than PersonA's ctor.
	// Example: p4's construction is more effective than p3,
	// After p4's construction, `s` is hollowed out, whose content moves to p4.
	//
	PersonB p4{ std::move(s) };
	
	return 0;
}
