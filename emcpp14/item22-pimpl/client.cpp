#include "widget.h"

void do_client()
{
	Widget w1;

	Widget w2 = w1;
}

int main(int argc, char *argv[])
{
	do_client();
	return 0;
}
