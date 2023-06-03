#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>

/*
 * Original code from [CPPLANG4] Bjarne Stroustrup, p361~362, applying Jimm's fix.
 
  [Visual C++ 2019]

	cl /EHsc /DCURRENT_MODE=Mode::throw_     ch13_bjassert.cpp  /link /out:bjthrow.exe
	cl /EHsc /DCURRENT_MODE=Mode::terminate_ ch13_bjassert.cpp  /link /out:bjterminate.exe
	cl /EHsc /DCURRENT_MODE=Mode::ignore_    ch13_bjassert.cpp  /link /out:bjignore.exe

  [gcc 7.5]
	 g++ -o bjthrow     -DCURRENT_MODE=Mode::throw_ ch13_bjassert.cpp
	 g++ -o bjterminate -DCURRENT_MODE=Mode::terminate_ ch13_bjassert.cpp
	 g++ -o bjignore    -DCURRENT_MODE=Mode::ignore_ ch13_bjassert.cpp

  [Run]
	bjthrow 3      // [Success]
	bjterminate 3  // [Success]
	bjthrow 4      // [Caught!]
	bjterminate 4  // Crash!
	bjignore 4     // [Success]
*/

using namespace std;

const int Max = 4;

#ifndef CURRENT_MODE
#define CURRENT_MODE Mode::throw_
#endif

#ifndef CURRENT_LEVEL
#define CURRENT_LEVEL 1
#endif

namespace Assert
{
	enum class Mode { throw_, terminate_, ignore_ };

	constexpr Mode current_mode = CURRENT_MODE;

	constexpr int current_level = CURRENT_LEVEL; // sneak level threshold
	constexpr int default_sneak_level = 1; // bjarne-name: default_level (not vivid)

	constexpr bool is_report_error(int sneak_level)
	{
		return sneak_level <= current_level; 
	}

	struct Error : runtime_error
	{
		Error(const string& p) : runtime_error(p) {}
	};

	string compose(const char* file, int line, const std::string& message)
		// compose message including file name and line number
	{
		ostringstream os;
		os << "(" << file << "," << line << "): " << message;
		return os.str();
	}

	template<
		bool is_report_error = is_report_error(default_sneak_level),
		class Except = Error>
	void dynamic(bool assertion, 
		const string& message="Assert::dynamic() failed.")
	{
		if (assertion)
			return; // runtime stuff OK, proceed 

		if (current_mode == Mode::throw_)
			throw Except{ message };

		if (current_mode == Mode::terminate_)
			std::terminate();

		if (current_mode == Mode::ignore_)
			true; // do nothing
	}
	
	template<>
	void dynamic<false, Error>(bool, const string&) // do nothing
	{		
	}
	
	void dynamic(bool runtime_stuff, const string& s) // default action
	{
		dynamic<true, Error>(runtime_stuff, s);
	}
	
	void dynamic(bool runtime_stuff) // default message
	{
		dynamic<true, Error>(runtime_stuff);
	}
}

void f(int n)
{
	// n should be in range [1:Max)
		
	Assert::dynamic(
		(n >= 1 && n < Max), 
		Assert::compose(__FILE__, __LINE__, "range problem"));
}

int main(int argc, char* argv[])
{
	int inputn = argc > 1 ? atoi(argv[1]) : 0;
	cout << "Calling f(" << inputn << ") ...\n";
	
	try
	{
		f(inputn);
		cout << "[Success]\n";
	}
	catch (std::exception &e)
	{
		cout << "[Caught!] " << e.what() << endl;
	}		

	return 0;
}

