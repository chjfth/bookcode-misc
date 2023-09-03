#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>

#include "chjdbg.h"

uint64 ps_GetOsMicrosecs64(void)
{
	LARGE_INTEGER li = {};
	QueryPerformanceCounter(&li);
	return li.QuadPart / 10;
}

const char *us_to_msecstring(uint64 usec)
{
	static char s_buf[40];

	_snprintf_s(s_buf, _TRUNCATE, "%d.%03d", int(usec/1000), int(usec%1000));
	return s_buf;
}
