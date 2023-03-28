#include <stdio.h>
#include <utility>

class Stringd
{
public:
	std::string ms;

public:
	Stringd() : ms("")
	{
		printf("<%p>Stringd-ctor()\n", this);
	}

	Stringd(const char *s) : ms(s)
	{
		printf("<%p>Stringd-ctor(\"%s\")\n", this, s);
	}

	Stringd(const Stringd& ins) : ms(ins.ms)
	{
		printf("<%p>Stringd-copy-ctor(\"%s\")   source<%p>\n", this, ms.c_str(), &ins);
	}

	Stringd(Stringd&& ins) : ms(std::move(ins.ms))
	{
		printf("<%p>Stringd-move-ctor(\"%s\")   emptying<%p>\n", this, ms.c_str(), &ins);
	}

	Stringd& operator=(const Stringd& ins)
	{
		printf("<%p>Stringd-copy-assign(\"%s\")   source<%p>\n", this, ins.ms.c_str(), &ins);
		this->ms = ins.ms;
		return *this;
	}

	Stringd& operator=(Stringd&& ins)
	{
		printf("<%p>Stringd-move-assign(\"%s\")  emptying<%p>\n", this, ins.ms.c_str(), &ins);
		this->ms = std::move(ins.ms);
		return *this;
	}

	~Stringd()
	{
		printf("<%p>Stringd-dtor(\"%s\")\n", this, ms.c_str());
	}

/*	operator const std::string& ()
	{
		return ms;
	}
*/
	operator std::string& ()
	{
		return ms;
	}
};
