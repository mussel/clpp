#include "clpp/StopWatch.h"

#ifdef WIN32

double StopWatch::LIToSecs(LARGE_INTEGER& L)
{
	return ((double)L.QuadPart /(double)frequency.QuadPart);
}
 
StopWatch::StopWatch()
{
	timer.start.QuadPart = 0;
	timer.stop.QuadPart = 0; 
	QueryPerformanceFrequency(&frequency);
}
 
void StopWatch::StartTimer()
{
	QueryPerformanceCounter(&timer.start);
}
 
void StopWatch::StopTimer()
{
	QueryPerformanceCounter(&timer.stop);
}
 
double StopWatch::GetElapsedTime()
{
	LARGE_INTEGER time;
	time.QuadPart = timer.stop.QuadPart - timer.start.QuadPart;
	return 1000 * LIToSecs(time);
}

#endif

#if defined(__linux__) || defined(__APPLE__)

StopWatch::StopWatch()
{
	// timer.start.QuadPart = 0;
	// timer.stop.QuadPart = 0; 
	// QueryPerformanceFrequency(&frequency);
}

void StopWatch::StartTimer( )
{

#ifdef __MACH__
 timestart = mach_absolute_time();
#else
  gettimeofday(&start,NULL);
#endif

}
 
void StopWatch::StopTimer()
{

#ifdef __MACH__
  timestop = mach_absolute_time();
#else
  gettimeofday(&end, NULL);
#endif


}

double StopWatch::GetElapsedTime()
{

#ifdef __MACH__
	mach_timebase_info_data_t info;
	mach_timebase_info(&info);

     uint64_t duration = timestop-timestart	;

     duration *= info.numer;
     duration /= info.denom;

     return (double) duration * 0.000001 ;

#else
	double elapsed = (end.tv_sec -start.tv_sec)*1000.0;
	elapsed += (end.tv_usec - start.tv_usec)/1000.0;
	return elapsed;
#endif
}

#endif
