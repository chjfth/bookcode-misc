#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "stack2.hpp" // p32
#include "stackpartspec.hpp" // p33

namespace p32
{
	void main()
	{
		Stack<int>         intStack;       // stack of ints
		Stack<std::string> stringStack;    // stack of strings

		// manipulate int stack
		intStack.push(7);
		std::cout << intStack.top() << '\n';

		// manipulate string stack
		stringStack.push("hello");
		std::cout << stringStack.top() << '\n';
		stringStack.pop();
	}
}

namespace p33
{
	void test()
	{
		Stack<int*> ptrStack; // stack of pointers (special implementation)
		ptrStack.push(new int{ 42 });
		std::cout << *ptrStack.top() << '\n';
		delete ptrStack.pop();
	}
}

int main()
{
	p33::test();
//	p32::main();
	
	return 0;
}

