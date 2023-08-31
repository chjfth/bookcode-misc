#ifndef __chjdbg_h_
#define __chjdbg_h_

extern"C"{

#ifdef _MSC_VER
#define int64 __int64
#else
#define int64 long long
#endif

unsigned int64 ps_GetOsMillisec64(void);
	// Get number of milliseconds since the device booted, as timing reference.


} // extern"C"

#endif