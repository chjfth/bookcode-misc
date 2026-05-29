#include <new>
#include <utility>
#include <string>
#include <stdio.h>

class WidgetF1   // from p169
{
public:
	template<typename T>
	void setName(T&& newName) // universal reference
	{
		name = std::forward<T>(newName);
	}

private:
	std::string name;
};


class WidgetD1   // from p170, dual setName() overloads
{
public:
	void setName(const std::string& newName) // set from const lvalue
	{
		name = newName;
	}

	void setName(std::string&& newName) // set from rvalue
	{
		name = std::move(newName);
	}
private:
	std::string name;
};


int main(int argc, char *argv[])
{
	const char myname[] = "Jimm";
	const char *pname = "Chen";

	WidgetF1 f1obj;
	f1obj.setName("Adela Novak"); // const char[12] &
	f1obj.setName(myname);        // const char[5] &
	f1obj.setName(pname);         // const char *&

	WidgetD1 d1obj;
	d1obj.setName("Adela Novak"); // set from rvalue: string&&
	d1obj.setName(myname);        // set from rvalue: string&&
	d1obj.setName(pname);         // set from rvalue: string&&
	
	std::string strobj("bigball");
	d1obj.setName(strobj); // set from const lvalue: const string&

	return 0;
}
