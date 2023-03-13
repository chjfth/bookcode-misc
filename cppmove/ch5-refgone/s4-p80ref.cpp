#include <string>
#include <stdio.h>

// Code adapted from [cppmove] p80

class Person
{
public:
	std::string name;
	const std::string& getName() const
	{
		return this->name; 
	}
};	

Person returnPersonByValue(const char *name)
{
	Person person = Person{name};
	return person;
}

void test_liveperson()
{
	const std::string& tomname = returnPersonByValue("Tom").getName();
	
	for(auto pos=tomname.begin(), end=tomname.end(); pos!=end; ++pos)
	{
		printf("> %c\n", *pos);
	}
	
	printf("\n");

	for(char c : returnPersonByValue("Bob").getName())
	{
		printf("> %c\n", c);
	}
}

int main()
{
	test_liveperson();
	return 0;
}
