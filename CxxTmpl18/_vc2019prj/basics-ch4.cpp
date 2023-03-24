#include <stdio.h>
#include <iostream>

namespace p55
{
	void print ()
	{
		std::cout << "[DEBUG] null-param print()\n";
	}
	
	template<typename T, typename... Types>
	void print (T firstArg, Types... args)
	{
		std::cout << firstArg << '\n'; // print first argument

		print(args...); // call print() for remaining arguments
	}

	void test()
	{
		std::string s("world");
		print (7.5, "hello", s);
	}
}

namespace p57
{
	template<typename T>
	void print (T arg)
	{
		std::cout << arg << '\n'; // print passed argument
	}
	
	template<typename T, typename... Types>
	void print (T firstArg, Types... args)
	{
		print(firstArg); // call print() for the first argument
		print(args...);  // call print() for remaining arguments
	}

	void test()
	{
		std::string s("world");
		print (7.5, "hello", s, "Again1", "Again2");
	}
}


int main()
{
	p57::test();
	
	return 0;
}

