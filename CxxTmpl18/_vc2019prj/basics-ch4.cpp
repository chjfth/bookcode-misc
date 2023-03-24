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


int main()
{
	p55::test();
//	p32::main();
	
	return 0;
}

