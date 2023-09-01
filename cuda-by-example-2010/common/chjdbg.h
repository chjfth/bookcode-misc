#ifndef __chjdbg_h_
#define __chjdbg_h_

extern"C"{

#ifdef _MSC_VER
#define int64 __int64
#else
#define int64 long long
#endif

unsigned int64 ps_GetOsMicrosecs64(void);
	// Get number of microseconds since the device booted, as timing reference.


const char *us_to_msecstring(unsigned int64 usec);
	// Microsec to millisec string.

} // extern"C"

#endif