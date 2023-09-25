#include <stdio.h>
#include <stdlib.h>
#include <tchar.h>
#include <locale.h>

extern"C" void main_what_is_my_id_kpB(int argc, char* argv[]);

int main(int argc, char* argv[])
{
	setlocale(LC_ALL, "");
	
	printf("Hello, what_is_my_id_kpB!\n");

	main_what_is_my_id_kpB(argc, argv);

	return 0;
}

