#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "stack2.hpp"

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

int main()
{
	p32::main();
	
	return 0;
}

