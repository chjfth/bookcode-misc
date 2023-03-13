#include <iostream>
#include <string>
#include <stdio.h>

// Code adapted from [cppmove] p80

class Person
{
public:
	std::string name;
	const std::string& getName() const
	{
		printf("In Person.getName(),    this = %p\n", this);
		return this->name; 
	}

	Person(const Person &person) 
	{
		this->name = person.name;
		printf("Person ctor([Person:\"%s\"]).   %p+\n", 
			person.name.c_str(), this);
	}
	Person(const char *name) 
	{
		this->name = name;
		printf("Person ctor(\"%s\").            %p+\n", 
			name, this);
	}
	~Person() 
	{
		printf("Person dtor().                 %p-\n", this);
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
	printf("                    &tomname = %p\n", &tomname);
	
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
