#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <deque>

using namespace std;

void p230_lambda()
{
	deque<int> coll = { 1, 3, 19, 5, 13, 7, 11, 2, 17 };
	
	int x = 5, y = 12;
	auto pos = find_if(coll.cbegin(), coll.cend(), // range
		[=] (int i)
		{
			return i > x && i < y;
		}
	);

	cout << "First elem >5 and <12: " << *pos << endl;
}

namespace p234 // class object as function-object
{
	class PrintInt
	{
	public:
		void operator() (int elem) const
		{
			cout << elem << ' ';
		}
	};

	void main()
	{
		vector<int> coll;

		// insert elements from 1 to 9
		for(int i=1; i<=9; ++i)
			coll.push_back(i);

		for_each(coll.cbegin(), coll.cend(), // range
			PrintInt());
		cout << endl;
	}
}



int main(int argc, char *argv[])
{
	p234::main();
	
	p230_lambda();
}
