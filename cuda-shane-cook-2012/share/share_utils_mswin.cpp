#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h> // For Desktop Windows, _beginthreadex

#include "share.h"

uint64 ps_GetOsMicrosecs64(void)
{
	LARGE_INTEGER li = {};
	QueryPerformanceCounter(&li);
	return li.QuadPart / 10;
}

struct SwThreadParam
{
	PROC_ggt_simple_thread proc;
	void *param;
};

static unsigned __stdcall	// PC Windows
_WinThreadWrapper(void * param)
{
	SwThreadParam *pw = (SwThreadParam*)param;

	pw->proc(pw->param);
	delete pw;

	return 0;
}


GGT_HSimpleThread
ggt_simple_thread_create(PROC_ggt_simple_thread proc, void *param, int stack_size)
{
	HANDLE hThread = NULL;
	SwThreadParam *pwp = new SwThreadParam;
	if(!pwp)
		return NULL;

	pwp->proc = proc, pwp->param = param;

	unsigned int tid = 0;
	hThread = (HANDLE)_beginthreadex(NULL, stack_size, _WinThreadWrapper, pwp ,0, &tid);

	return (GGT_HSimpleThread)hThread;
}

bool 
ggt_simple_thread_waitend(GGT_HSimpleThread h)
{
	if(!h) {
		return false;
	}

	DWORD waitre = WaitForSingleObject(h, INFINITE);
	if(waitre==WAIT_OBJECT_0)
	{
		CloseHandle(h);
		return true;
	}
	else
		return false;
}
