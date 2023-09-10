#ifndef __chjdbg_h_
#define __chjdbg_h_

extern"C"{

#ifdef _MSC_VER
#define int64 __int64
#else
#define int64 long long
#endif

typedef unsigned int64 uint64;

uint64 ps_GetOsMicrosecs64(void);
	// Get number of microseconds since the device booted, as timing reference.


const char *us_to_msecstring(uint64 usec);
	// Microsec to millisec string.

void dump_microseconds_diffs(unsigned int ar_uints[], int arsize);

#ifdef _MSC_VER
#define C_SNPRINTF _snprintf_s
#else // For Linux
#define C_SNPRINTF snprintf
#endif


} // extern"C"

#endif