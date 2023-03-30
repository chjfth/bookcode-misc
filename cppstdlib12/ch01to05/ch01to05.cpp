#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>

#include "Stringd.h"

using namespace std;

namespace p77
{
	void main()
	{
		// two shared pointers representing two persons by their name
		shared_ptr<string> pNico(new string("nico"));
		shared_ptr<string> pJutta(new string("jutta"));

		// capitalize person names
		(*pNico)[0] = 'N';
		pJutta->replace(0, 1, "J");
		
		// put them multiple times in a container
		vector<shared_ptr<string>> whoMadeCoffee;
		whoMadeCoffee.push_back(pJutta);
		whoMadeCoffee.push_back(pJutta);
		whoMadeCoffee.push_back(pNico);
		whoMadeCoffee.push_back(pJutta);
		whoMadeCoffee.push_back(pNico);

		// print all elements
		for (auto ptr : whoMadeCoffee) {
			cout << *ptr << " ";
		}
		cout << endl;

		// overwrite a name again
		*pNico = "Nicolai";

		// print all elements again
		for (auto ptr : whoMadeCoffee) {
			cout << *ptr << " ";
		}
		cout << endl;
		// print some internal data
		cout << "use_count: " << whoMadeCoffee[0].use_count() << endl;
	}

	void main_p79_delete_Nico_early()
	{
		// two shared pointers representing two persons by their name
		shared_ptr<Stringd> pNico(new Stringd("nico"));
		shared_ptr<Stringd> pJutta(new Stringd("jutta"));

		// capitalize person names
		(*pNico).ms[0] = 'N';
		pJutta->ms.replace(0, 1, "J");

		// put them multiple times in a container
		vector<shared_ptr<Stringd>> whoMadeCoffee;
		whoMadeCoffee.push_back(pJutta);
		whoMadeCoffee.push_back(pJutta);
		whoMadeCoffee.push_back(pNico);
		whoMadeCoffee.push_back(pJutta);
		whoMadeCoffee.push_back(pNico);

		// print all elements
		for (auto ptr : whoMadeCoffee) {
			cout << (*ptr).ms << " ";
		}
		cout << endl;

		// overwrite a name again
		(*pNico).ms = "Nicolai";

		// print all elements again
		for (auto ptr : whoMadeCoffee) {
			cout << (*ptr).ms << " ";
		}
		cout << endl;
		// print some internal data
		cout << "pJutta use_count: " << whoMadeCoffee[0].use_count() << endl;
		pJutta = nullptr;
		cout << "After pJutta=nullptr, use_count: " << whoMadeCoffee[0].use_count() << endl;

		pNico = nullptr;
		whoMadeCoffee.resize(2);
		
		cout << "main_p79_delete_Nico_early() now return.\n";
	}
}

namespace p80_user_deleter
{
	void main()
	{
		// two shared pointers representing two persons by their name
		shared_ptr<string> pNico(new string("nico"), 
			[](string* p)
			{
				cout << "Custom delete: " << *p << endl;
			});
		shared_ptr<string> pJutta(new string("jutta"));

		// capitalize person names
		(*pNico)[0] = 'N';
		pJutta->replace(0, 1, "J");

		// put them multiple times in a container
		vector<shared_ptr<string>> whoMadeCoffee;
		whoMadeCoffee.push_back(pJutta);
		whoMadeCoffee.push_back(pJutta);
		whoMadeCoffee.push_back(pNico);
		whoMadeCoffee.push_back(pJutta);
		whoMadeCoffee.push_back(pNico);

		// print all elements
		for (auto ptr : whoMadeCoffee) {
			cout << *ptr << " ";
		}
		cout << endl;

		// overwrite a name again
		*pNico = "Nicolai";

		// print all elements again
		for (auto ptr : whoMadeCoffee) {
			cout << *ptr << " ";
		}
		cout << endl;
		// print some internal data
		cout << "use_count: " << whoMadeCoffee[0].use_count() << endl;

		// p80: Now retire pNico early.
		pNico = nullptr;
		whoMadeCoffee.resize(2);
		
		cout << "p80_user_deleter::main() now return.\n";
	}
}

namespace p81_orig 
{
	class FileDeleter // util/sharedptr2.cpp
	{
	private:
		std::string filename;
	public:
		FileDeleter (const std::string& fn)
			: filename(fn) {
		}
		void operator () (std::ofstream* fp) {
			fp->close(); // close.file
			std::remove(filename.c_str()); // delete file
		}
	};

	void main(const char *prg="")
	{
		// create and open temporary file
		std::shared_ptr<std::ofstream> fp(
			new std::ofstream("tmpfile.txt"),
			FileDeleter("tmpfile.txt")
		);
	}
}

namespace p81_mod1
{
	class FileDeleter
	{
	private:
		std::string filename;
	public:
		FileDeleter (const std::string& fn)
			: filename(fn) {
		}

		~FileDeleter() {
			cout << "~FileDeleter() called.\n";
		}
		
		void operator () (std::ofstream* fp) {
			fp->close(); // close.file

			//std::remove(filename.c_str()); // don't delete, keep for investigate
		}
	};

	void main(const char* prg = "")
	{
		// create and open temporary file

		const char* myfile = "tmpfile.txt";
		auto ofile = new std::ofstream(myfile);
		std::shared_ptr<std::ofstream> fp(ofile, FileDeleter(myfile));

		(*fp) << "p81 main(): " << prg << endl;
	}
}


int main(int argc, char *argv[])
{
	p80_user_deleter::main();
	p81_mod1::main(argv[0]);
}
