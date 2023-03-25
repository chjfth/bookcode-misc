#include <stdio.h>
#include <complex>
#include <iostream>
#include <tuple>
#include <vector>
#include <array>

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

namespace p58
{
	void print () // p58: Need this to compile success.
	{
		std::cout << "[DEBUG] null-param print()\n";
	}

	template<typename T, typename... Types>
	void print (T firstArg, Types... args)
	{
		std::cout << firstArg << '\n';
		if (sizeof...(args) > 0) { // error if sizeof...(args)==0
			print(args...); // and no print() for no arguments declared
		}
	}

	void test()
	{
		print ("p58 COMPILE ERROR\n");
	}
}

namespace p134 // compared to p58, now using constexpr
{
	template<typename T, typename... Types>
	void print (T firstArg, Types... args)
	{
		std::cout << firstArg << '\n';
		if constexpr (sizeof...(args) > 0) {
			print(args...); // code only available if sizeof...(args)>0 (since C++17)
		}
	}

	void test()
	{
		std::string s("world");
		print ("p134:", "hello", s);
	}
}

namespace p59
{
	// basics/foldtraverse.cpp
	// define binary tree structure and traverse helpers:
	struct Node {
		int value;
		Node* left;
		Node* right;
		Node(int i = 0) : value(i), left(nullptr), right(nullptr) {	}
	};
	
	auto left = &Node::left;
	auto right = &Node::right;

	// traverse tree, using fold expression:
	template<typename T, typename... TP>
	Node* traverse (T np, TP... paths) {
		return (np ->* ...->*paths); // np ->* paths1 ->* paths2 ...
	}

	void main()
	{
		// init binary tree structure:
		Node* root = new Node{ 0 };
		root->left = new Node{ 1 };
		root->left->right = new Node{ 2 };

		// traverse binary tree:
		Node* node = traverse(root, left, right);
		std::cout << "p59: Final node value is: " << node->value << '\n';
	}
}

namespace p60
{
	template<typename... Types>
	void print0 (Types const&... args)
	{
		(std::cout << ... << args) << '\n';
	}

	void test0()
	{
		std::string s("world");
		print0 ("p60pre:", "hello", s); // Will print "p60pre:helloworld"
	}

	// basics/addspace.hpp
	template<typename T>
	class AddSpace
	{
	private:
		T const& ref; // refer to argument passed in constructor
	public:
		AddSpace(T const& r) : ref(r) {
		}
		
		friend std::ostream& operator<< (std::ostream& os, AddSpace<T> s) {
			return os << s.ref << ' '; // output passed argument and a space
		}
	};

	template<typename... Args>
	void print (Args... args) {
		( std::cout << ... << AddSpace(args) ) << '\n';
	}

	void test()
	{
		std::string s("world");
		print ("p60:", "hello", s); // Will print "p60:helloworld"
	}
}

namespace ch4
{
	template<typename T, typename... Types>
	void print (T firstArg, Types... args)
	{
		std::cout << firstArg << '\n';
		if constexpr (sizeof...(args) > 0) {
			print(args...); // code only available if sizeof...(args)>0 (since C++17)
		}
	}
}

namespace p62
{
	using namespace ch4;
	
	template<typename... T>
	void printDoubled (T const&... args)
	{
		print (args + args...);
	}

	void test()
	{
		printDoubled(7.5, std::string("hello"), std::complex<float>(4, 2));
	}
}

namespace p63
{
	// uses a variadic list of indices to access the
	// corresponding element of the passed first argument

	using namespace ch4;
	
	template<typename C, typename... Idx>
	void printElems (C const& coll, Idx... idx)
	{
		print (coll[idx]...);
	}

	template<std::size_t... Idx, typename C>
	void printIdx (C const& coll)
	{
		print(coll[Idx]...);
	}
	
	void test()
	{
		std::vector<std::string> coll = { "good", "times", "say", "bye" };
		printElems(coll, 2, 0, 3);

		std::cout << "====\n";
		printIdx<2, 0, 3>(coll);
	}
}

namespace p64
{
	using namespace ch4;

	// type for arbitrary number of indices:
	template<std::size_t...>
	struct Indices {};

	template<typename T, std::size_t... Idx>
	void printByIdx(T t, Indices<Idx...>) // book p64
	{
		print(std::get<Idx>(t)...);
	}

	void test()
	{
		std::array<std::string, 5> arr = { "Hello", "my", "new", "!", "World" };
		printByIdx(arr, Indices<0, 4, 3>());
		
		auto t = std::make_tuple(12, "monkeys", 2.0);
		printByIdx(t, Indices<0, 1, 2>());
	}

	template<typename T, typename... TIdx>
	void printByIdx_me(T t, TIdx... Idx)
	{
		print(t[Idx]...);
	}

	void test_me()
	{
		std::array<std::string, 5> arr = { "Hello", "my", "new", "!", "World" };
		printByIdx_me(arr, 0, 4, 3);
	}
}

int main()
{
	p64::test_me();
//	p59::main();
	
	return 0;
}

