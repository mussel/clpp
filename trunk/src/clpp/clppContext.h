#ifndef __CLPP_CONTEXT_H__
#define __CLPP_CONTEXT_H__

#if defined (__APPLE__) || defined(MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

class clppContext
{
public:
	cl_context clContext;			// OpenCL context
	cl_device_id clDevice;			// OpenCL Device
	cl_command_queue clQueue;		// OpenCL command queue 

	// Default setup : use the default platform and default device
	void setup();
};

#endif