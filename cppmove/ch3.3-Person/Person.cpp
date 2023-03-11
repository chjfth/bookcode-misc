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
		coll.push_back(p);            // [B0] OK, copies p
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
		Person& operator=(const Person&) = delete;
		
		// NO move constructor/assignment declared
	};

	void test_p50()
	{
		std::vector<Person> coll;

		Person p{ "Tina", "fox" };
		
		coll.push_back(p);            // [B3] OK, copies p
		coll.push_back(std::move(p)); // [D3] OK, copies p (as fallback)
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
		
		// move constructor/assignment positively declared:
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


namespace p52a
{
	// p52a: Delete both copy-ctor and move-ctor.

	class Person
	{
	public:
		std::string sur, giv;
		Person(const char* s, const char* g)
			: sur{ s }, giv{ g } {}

	public:
		// NO copy constructor declared.

		// move constructor/assignment declared as deleted:
		Person(Person&&) = delete; // ! delete, diff to p51
		Person& operator=(Person&&) = delete;
	};

	void test_p52a()
	{
		std::vector<Person> coll;

		Person p{ "Tina", "fox" };

#ifdef SEE_ERROR_p52a1
		coll.push_back(p);            // [B5] ERROR: copying implicitly deleted
#endif
#ifdef SEE_ERROR_p52a2
		coll.push_back(std::move(p)); // [D5] ERROR: moving explicitly deleted
#endif
	}
}


namespace p52b
{
	// p52b: Delete both copy-ctor and move-ctor. (better)

	class Person
	{
	public:
		std::string sur, giv;
		Person(const char* s, const char* g)
			: sur{ s }, giv{ g } {}

	public:
		// Copy constructor/assignment declared as deleted:
		Person(const Person&) = delete; // ! delete, diff to p50
		Person& operator=(const Person&) = delete;

		// Chj: Leave move-ctor/assignment as "default" proposal,
		// which fallback to copy-ctor/assignment,
		// resulting in "deleted" as well.
	};

	void test_p52b()
	{
		std::vector<Person> coll;

		Person p{ "Tina", "fox" };

#ifdef SEE_ERROR_p52b1
		coll.push_back(p);            // [B5] ERROR: copying disabled
#endif
#ifdef SEE_ERROR_p52b2
		coll.push_back(std::move(p)); // [D5] ERROR: moving disabled
#endif
	}
}


namespace p52c
{
	// p52c: Delete move-ctor but enable copy-ctor, OK but no actual sense.

	class Person
	{
	public:
		std::string sur, giv;
		Person(const char* s, const char* g)
			: sur{ s }, giv{ g } {}

	public:
		// copy constructor positive declared:
		Person(const Person& p) = default;
		Person& operator=(const Person&) = default;

		// move constructor/assignment declared as deleted:
		Person(Person&&) = delete;
		Person& operator=(Person&&) = delete;
	};

	void test_p52c()
	{
		std::vector<Person> coll;

		Person p{ "Tina", "fox" };

#ifdef SEE_ERROR_p52c_vc
		// VC2019 ERROR on this, but gcc-12 not.
		coll.push_back(p);            // [B3] OK(gcc): copying enabled
#endif
#ifdef SEE_ERROR_p52c2
		coll.push_back(std::move(p)); // [D5] ERROR: moving disabled
#endif
	}
}

namespace p53c
{
	class Person
	{
	public:
		std::string sur, giv;
		Person(const char* s, const char* g)
			: sur{ s }, giv{ g } {}

	public:
		// Omit copy constructor/assignment.
		// Omit move constructor/assignment.

		// Add own ctor!
		// Result: move-ctor fallbacks to copy-ctor.
		Person() { }
	};

	void test_p53c()
	{
		std::vector<Person> coll;

		Person p{ "Tina", "fox" };

		coll.push_back(p);            // [B0] OK, copies p (default copy-tor)
		coll.push_back(std::move(p)); // [D0] OK, moves p  (default move-tor)
	}
}


namespace p53d
{
	class Person
	{
	public:
		std::string sur, giv;
		Person(const char* s, const char* g)
			: sur{ s }, giv{ g } {}

	public:
		// Omit copy constructor/assignment.
		// Omit move constructor/assignment.

		// Add own dtor!
		// Result: move-ctor fallbacks to copy-ctor.
		~Person(){ }
	};

	void test_p53d()
	{
		std::vector<Person> coll;

		Person p{ "Tina", "fox" };

		coll.push_back(p);            // [B7] OK, copies p
		coll.push_back(std::move(p)); // [D7] OK, copies p (as fallback)
	}
}

namespace p53cd
{
	class Person
	{
	public:
		std::string sur, giv;
		Person(const char* s, const char* g)
			: sur{ s }, giv{ g } {}

	public:
		// Omit copy constructor/assignment.
		// Omit move constructor/assignment.

		// Add own ctor & dtor!
		// Result: move-ctor fallbacks to copy-ctor.
		Person() { }
		~Person() { }
	};

	void test_p53cd()
	{
		std::vector<Person> coll;

		Person p{ "Tina", "fox" };

		coll.push_back(p);            // ? [B7] OK, copies p
		coll.push_back(std::move(p)); // ? [D7] OK, copies p (as fallback)
	}
}



int main(int argc, char* argv[])
{
	printf("Hello, cppmove CH3.3 Person!\n");

	// NOTE to  user: You need to observe the program behavior in a debugger.

	p50z::test_p50z();

	p50::test_p50();

	p51::test_p51();

	p53c::test_p53c();
	p53d::test_p53d();
	p53cd::test_p53cd();

	return 0;
}

