#include <stdio.h>
#include <msvc_extras.h>
#include "widget_ensureclnup.h"

void do_client()
{
	Widget w1;

	Widget w2 = w1;
}

int main(int argc, char *argv[])
{
	MSVCRT_MemCheckStart(foo);

	do_client();

	//int* pi_leak = new int(0x11223344);

	bool isleak = MSVCRT_MemCheckEnd_IsLeak(foo);
	if (isleak)
		return 4;
	else
		return 0;
}
