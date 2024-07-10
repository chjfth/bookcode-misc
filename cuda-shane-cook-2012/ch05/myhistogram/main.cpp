#include <stdio.h>
#include <stdlib.h>
// #include <tchar.h>
#include <locale.h>

extern"C" void main_myhistogram(int argc, char* argv[]);

int main(int argc, char* argv[])
{
	setlocale(LC_ALL, "");
	
	main_myhistogram(argc, argv);

	return 0;
}

