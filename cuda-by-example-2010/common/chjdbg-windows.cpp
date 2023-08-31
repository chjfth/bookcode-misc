#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "chjdbg.h"

unsigned int64 
ps_GetOsMillisec64(void)
{
	return GetTickCount64();
}
