#include <iostream>
#include <string>
#include <string_view>

// From cpp17cg book p189

class Person {
	std::string name;
public:
	Person(const char *_name) : name(_name)
	{}
	
	std::string_view getName() const { // don’t do this
		return name;
	}
};

Person CreatePerson(const char *name)
{
	return Person(name);
}

int main(int argc, char *argv[])
{
	Person person1("Tim");

	auto n1 = person1.getName();
	std::cout << "name1: " << n1 << '\n'; // no problem

	auto n2 = CreatePerson("Tom").getName(); // Bad! n2 refers to temporal string.
	std::cout << "name2: " << n2 << '\n'; 
	
	return 0;
}
