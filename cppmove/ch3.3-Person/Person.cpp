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
		std::string sur, giv;
		
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
		coll.push_back(p);            // [C0] OK, copies p
		coll.push_back(std::move(p)); // [D0] OK, moves p	
	}

} // namespace p50z

namespace p50
{
	// The example from page 50.
	// Declared Copying Disables Moving (Fallback Enabled)
	
	class Person
	{
	public:
		std::string sur, giv;
		Person(const char* s, const char* g)
			: sur{s}, giv{g} {}

	public:
		// copy constructor/assignment declared:
		Person(const Person&) = default;
		Person& operator=(const Person&) = default;
		
		// NO move constructor/assignment declared
	};

	void test_p50()
	{
		std::vector<Person> coll;

		Person p{ "Tina", "fox" };
		
		coll.push_back(p);            // [C0] OK, copies p
		coll.push_back(std::move(p)); // [D3] OK, copies p (as fallback)
	}
	
} // namespace p50


namespace p50b
{
	// [Chj] Based on p50, plus: mark move-ctor =deleted.

	class Person
	{
	public:
		std::string sur, giv;
		Person(const char* s, const char* g)
			: sur{ s }, giv{ g } {}

	public:
		// copy constructor/assignment declared:
		Person(const Person&) = default;
		Person& operator=(const Person&) = default;

#ifdef SEE_ERROR_p50b1
		// Mark move-ctor deleted.
		Person(Person&&) = delete;
#endif
	};

	void test_p50b()
	{
		std::vector<Person> coll;

		Person p{ "Tina", "fox" };

		coll.push_back(p);            // [C0] OK, copies p
#ifdef SEE_ERROR_p50b1
		coll.push_back(std::move(p)); // ERROR compile.
#endif
	}

} // namespace p50



namespace p51
{
	// p51: Declared-moving deletes copying.

	class Person
	{
	public:
		std::string sur, giv;
		Person(const char* s, const char* g)
			: sur{ s }, giv{ g } {}

	public:
		// NO copy constructor declared.
		
		// move constructor/assignment declared:
		Person(Person&&) = default;
		Person& operator=(Person&&) = default; //Chj: writing `=delete` (same result here)
	};

	void test_p51()
	{
		std::vector<Person> coll;

		Person p{ "Tina", "fox" };

#ifdef SEE_ERROR_p51
		coll.push_back(p);            // [B5] ERROR: copying deleted
#endif
		coll.push_back(std::move(p)); // [D5] OK, moves p

		coll.push_back(Person{ "Ben", "Cook" }); // [D5]
	}
}

int main(int argc, char* argv[])
{
	printf("Hello, cppmove CH3.3 Person!\n");

	// NOTE to  user: You need to observe the program behavior in a debugger.

	p50z::test_p50z();

	p50::test_p50();

	p51::test_p51();

	return 0;
}

