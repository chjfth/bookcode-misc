#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>

namespace p50z 
{
	// The example from page 50.
	// By Default, We Have Copying(copy-ctor) and Moving(move-ctor)
	
	class Person
	{
	public:
		std::string sur;
		std::string giv;
		
	public:
		// NO copy constructor/assignment declared
		// NO move constructor/assignment declared
		// NO destructor declared
	};
	
	void test_p50z()
	{
		// In this case, a Person can be both copied and moved:

		std::vector<Person> coll;
		Person p { "Tina", "Fox" };
		coll.push_back(p);			  // [C0] OK, copies p
		coll.push_back(std::move(p)); // [D0] OK, moves p	
	}

} // namespace p50z


int main(int argc, char* argv[])
{
	setlocale(LC_ALL, "");
	
	printf("Hello, cppmove CH3.3 Person!\n");

	p50z::test_p50z();

	return 0;
}

