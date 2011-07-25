#ifndef __CLPP_STOPWATCH_H__
#define __CLPP_STOPWATCH_H__


#ifdef WIN32
#include <windows.h>
 
typedef struct
{
	LARGE_INTEGER start;
	LARGE_INTEGER stop;
} stopWatch;

#endif


#if defined(__linux__) || defined(__APPLE__)

#include <time.h>
#include <sys/time.h>

#endif

#ifdef __APPLE__
#include <stdint.h>
#include <mach/mach_time.h>
#endif



 
class StopWatch
{
private:
#ifdef WIN32
	stopWatch timer;
	LARGE_INTEGER frequency;
	double LIToSecs(LARGE_INTEGER& L);
#endif

#if defined(__linux__) || defined(__APPLE__)

   timeval start;
   timeval end;

#ifdef __APPLE__
   uint64_t  timestart;
   uint64_t  timestop;
#endif


#endif




public:
	StopWatch();

	void StartTimer();
	void StopTimer();

	double GetElapsedTime();
};

#endif
