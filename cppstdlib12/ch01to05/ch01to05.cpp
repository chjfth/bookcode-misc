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

namespace p78_make_shared
{
	using namespace std;
	void test()
	{
		shared_ptr<string> pNico1 = make_shared<string>("nico1");
		shared_ptr<string> pJutta1 = make_shared<string>("jutta1");

		shared_ptr<string> pNico2{ new string("nico2") };
		shared_ptr<string> pJutta2{ new string("jutta2") };

		printf("sizeof(pNico1)=%d , sizeof(&pNico1)=%d\n", 
			(int)sizeof(pNico1), (int)sizeof(&pNico1));

		printf("&pNico1  = %p\n", &pNico1);
		printf("&pJutta1 = %p\n", &pJutta1);
		printf("&pNico2  = %p\n", &pNico2);
		printf("&pJutta2 = %p\n", &pJutta2);

		printf("\n");
		printf("pNico1 - pJutta1: %d\n", int((char*)&pNico1 - (char*)&pJutta1));
		printf("pNico2 - pJutta2: %d\n", int((char*)&pNico2 - (char*)&pJutta2));

	}

	void test_shared_ptr_2refs()
	{
		string* pstring = new string("ABC");
		shared_ptr<string> sp1{ pstring };
		shared_ptr<string> sp2 = sp1;

		printf("sizeof(string)=%d , sizeof(share_ptr<string>)=%d\n",
			(int)sizeof(string), (int)sizeof(shared_ptr<string>));

		int* sp1raw = (int*)&sp1;
		int* sp2raw = (int*)&sp2;
		printf("sp1 dump 2 ints: 0x%08X , 0x%08X\n", sp1raw[0], sp1raw[1]);
		printf("sp2 dump 2 ints: 0x%08X , 0x%08X\n", sp2raw[0], sp2raw[1]);
	}
}

namespace p80_custom_deleter
{
	void main()
	{
		// two shared pointers representing two persons by their name
		shared_ptr<string> pNico(new string("nico"), 
			[](string* p) // lambda deleter
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

namespace p81_FileDeleter_orig 
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
			delete fp;					   // close.file
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
			delete fp; // close file

			// Not calling std::remove(), bcz I want to keep file content for investigation
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
		shared_ptr<Person> mom(new Person(name + "'s mom"));
		shared_ptr<Person> dad(new Person(name + "'s dad"));
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

namespace p87
{
	class Person { // util/weakptr2.cpp
	public:
		string name;
		shared_ptr<Person> mother;
		shared_ptr<Person> father;
		vector<weak_ptr<Person>> kids; // using weak pointer !

		Person (const string& n,
			shared_ptr<Person> m = nullptr,
			shared_ptr<Person> f = nullptr)
			: name(n), mother(m), father(f) {
		}
		~Person() {
			cout << "delete " << name << endl;
		}
	};

	shared_ptr<Person> initFamily (const string& name)
	{
		shared_ptr<Person> mom(new Person(name + "'s mom"));
		shared_ptr<Person> dad(new Person(name + "'s dad"));

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
			<< p->mother->kids[0].lock()->name << endl;

		p = initFamily("jim");
		cout << "jim's family exists" << endl;
	}

	void test_empty_weakptr() 
	{
		shared_ptr<Person> p = initFamily("nico");
		cout << "nico's family exists" << endl;
		cout << "- nico is shared " << p.use_count() << " times" << endl;
		cout << "- name of 1st kid of nico's mom: "
			<< p->mother->kids[0].lock()->name << endl;

		shared_ptr<Person> mom1 = p->mother;

		p = initFamily("jim");
		cout << "jim's family exists" << endl;

		// ==== Chj test ====
		auto kid0  = mom1->kids[0];
		auto nico0 = mom1->kids[0].lock();
		cout << " # Chj checking nico-mon's kid again: " << nico0 << endl;

		/* VC2019 output:
nico's family exists
- nico is shared 1 times
- name of 1st kid of nico's mom: nico
delete nico
delete nico's dad
jim's family exists
 # Chj checking nico-mon's kid again: 0000000000000000
delete nico's mom
delete jim
delete jim's dad
delete jim's mom
		*/
	}
}

void test_weakptr()
{
	string* pstring = new string("abc");

	shared_ptr<string> sp1(pstring);
	weak_ptr<string> wp1(sp1);

	// Memo:
	//		&(*sp1._Rep)._Uses == &(*wp1._Rep)._Uses
	// So, the weak_ptr object's control block is exactly its master-shared_ptr's control block.

	weak_ptr<string> weak0;
	auto lock0 = weak0.lock();
	weak0 = sp1;
}

namespace p91 // enable_shared_from_this
{
	class Person : public enable_shared_from_this<Person> {
	public:
		string name;
		shared_ptr<Person> mother;
		shared_ptr<Person> father;
		vector<weak_ptr<Person>> kids;  // weak pointer !!!

		Person (const string& n)
			: name(n) {
		}

		void setParentsAndTheirKids (shared_ptr<Person> m = nullptr,
			shared_ptr<Person> f = nullptr)
		{
			mother = m;
			father = f;
			if (m != nullptr) {
				m->kids.push_back(shared_from_this());
			}
			if (f != nullptr) {
				f->kids.push_back(shared_from_this());
			}
		}

		~Person() {
			cout << "delete " << name << endl;
		}
	};

	shared_ptr<Person>
	initFamily (const string& name)
	{
		shared_ptr<Person> mom(new Person(name + "'s mom"));
		shared_ptr<Person> dad(new Person(name + "'s dad"));

		Person* pkid = new Person(name);
		shared_ptr<Person> kid(pkid);

		kid->setParentsAndTheirKids(mom, dad);
		return kid;
	}

	void main()
	{
		shared_ptr<Person> p = initFamily("nico");
		cout << "nico's family exists" << endl;
		cout << "- nico is shared " << p.use_count() << " times" << endl;
		cout << "- name of 1st kid of nico's mom: "
			<< p->mother->kids[0].lock()->name << endl;

		p = initFamily("jim");
		cout << "jim's family exists" << endl;
	}
}

#if 0
string&& foo ()
{
	string x;
	return x; // compile ERROR: returns reference to nonexisting object
}
#endif

void trivial_test()
{
	unique_ptr<string> up1 (new string("abc"));
	unique_ptr<string> up2 (new string("Abc"));
	up1.reset(new string("zyx"));
}

int main(int argc, char *argv[])
{
//	test_weakptr();
//	p91::main();
	
//	p78_make_shared::test();
//	p78_make_shared::test_shared_ptr_2refs();
	
//	p87::test_empty_weakptr();
//	p87::main();
//	p85_circref::main();
	
//	p77::main();
	
//	p81_mod1::test_reset(argv[0]);
	
//	p80_user_deleter::main();
	p81_mod1::main(argv[0]);
}
