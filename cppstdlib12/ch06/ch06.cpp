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

int main(int argc, char *argv[])
{
	p230_lambda();
}
