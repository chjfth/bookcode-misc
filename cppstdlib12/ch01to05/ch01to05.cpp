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
			cout << "+FileDeleter() +'" << fn << "'\n";
		}
		~FileDeleter() {
			cout << "~FileDeleter() '" << filename << "'\n";
		}
		void operator () (std::ofstream* fp) {
			fp->close(); // close file, but keep file content for investigation
		}

		FileDeleter(FileDeleter&& src)
		{
			cout << "+FileDeleter() move: '" << src.filename << "'\n";
			filename = std::move(src.filename);
		}
	};

	void main(const char* prg = "")
	{
		// create and open temporary file
		const char* myfile = "tmpfile.txt";
		auto ofile = new std::ofstream(myfile);
		std::shared_ptr<std::ofstream> spFile(ofile, FileDeleter(myfile));

		cout << "[strt] Write to file\n";
		(*spFile) << "p81 main(): " << prg << endl;
		cout << "[done] Write to file\n";
	}

	void test_reset(const char* prg = "")
	{
		// create and open temporary file
		const char* myfile = "tmpfile_p81mod.txt";
		auto ofile = new std::ofstream(myfile);
		std::shared_ptr<std::ofstream> spFile(ofile, FileDeleter(myfile));

		(*spFile) << "p81 test_reset(): " << prg << endl;

		spFile.reset();

		cout << "test_reset() return.\n";
	}
}

#ifdef __linux__ // __USE_POSIX
#include <memory> // for shared_ptr
#include <sys/mman.h> // for shared memory (linux only)
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring> // for strerror()
#include <cerrno> // for errno
#include <string>
#include <iostream>

namespace p82
{
	class SharedMemoryDetacher
	{
	public:
		void operator () (int* p)
		{
			std::cout << "unlink /tmp1234" << std::endl;
			if (shm_unlink("/tmp1234") != 0) {
				std::cerr << "OOPS: shm_unlink() failed" << std::endl;
			}
		}
	};
	
	std::shared_ptr<int> getSharedIntMemory (int num)
	{
		void* mem;
		int shmfd = shm_open("/tmp1234", O_CREAT | O_RDWR, S_IRWXU | S_IRWXG);
		if (shmfd < 0) {
			throw std::string(strerror(errno));
		}
		if (ftruncate(shmfd, num * sizeof(int)) == -1) {
			throw std::string(strerror(errno));
		}
		mem = mmap(nullptr, num * sizeof(int), PROT_READ | PROT_WRITE,
			MAP_SHARED, shmfd, 0);
		if (mem == MAP_FAILED) {
			throw std::string(strerror(errno));
		}
		return std::shared_ptr<int>(static_cast<int*>(mem), SharedMemoryDetacher());
	}
	
	void main()
	{
		// get and attach shared memory for 100 ints:
		std::shared_ptr<int> smp(getSharedIntMemory(100));
		
		// init the shared memory
		for (int i = 0; i < 100; ++i) {
			smp.get()[i] = i * 42;
		}

		// deal with shared memory somewhere else
		// ...

		std::cout << "<return>" << std::endl;

		// release shared memory here:
		smp.reset();
	}
}

#endif // __USE_POSIX

namespace p85_circref
{
	class Person {
	public:
		string name;
		shared_ptr<Person> mother;
		shared_ptr<Person> father;
		vector<shared_ptr<Person>> kids;

	Person (const string& n,
			shared_ptr<Person> m = nullptr,
			shared_ptr<Person> f = nullptr)
			: name(n), mother(m), father(f) {
		}
		~Person() {
			// MEMORY LEAK !!! 
			// The dtor never has a chance to be called in this program.
			cout << "delete " << name << endl;
		}
	};

	shared_ptr<Person> initFamily (const string& name)
	{
		shared_ptr<Person> mom(new Person(name + "’s mom"));
		shared_ptr<Person> dad(new Person(name + "’s dad"));
		shared_ptr<Person> kid(new Person(name, mom, dad));
		mom->kids.push_back(kid);
		dad->kids.push_back(kid);
		return kid;
	}

	void main()
	{
		shared_ptr<Person> p = initFamily("nico");
		cout << "nico's family exists" << endl;
		cout << "- nico is shared " << p.use_count() << " times" << endl;
		cout << "- name of 1st kid of nico's mom: "
			<< p->mother->kids[0]->name << endl;

		p = initFamily("jim");
		cout << "jim's family exists" << endl;
	}
}


int main(int argc, char *argv[])
{
	p85_circref::main();
	
//	p77::main();
	
//	p81_mod1::test_reset(argv[0]);
	
//	p80_user_deleter::main();
//	p81_mod1::main(argv[0]);
}
