#include "clpp/clppContext.h"

#include<assert.h>

void clppContext::setup()
{
	cl_int clStatus;

	//---- Retreive information about platforms
	cl_uint platformsCount;
	clStatus = clGetPlatformIDs(0, NULL, &platformsCount);
	assert(clStatus == CL_SUCCESS);

	cl_platform_id* platforms = new cl_platform_id[platformsCount];
	clStatus = clGetPlatformIDs(platformsCount, platforms, NULL);
	assert(clStatus == CL_SUCCESS);

	//---- Devices
	cl_uint devicesCount;
	clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &devicesCount);
	assert(clStatus == CL_SUCCESS);
	assert(devicesCount > 0);
	
	cl_device_id* devices = new cl_device_id[devicesCount];
	clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, devicesCount, &clDevice, NULL);
	assert(clStatus == CL_SUCCESS);

	//---- Context
	clContext = clCreateContext(0, 1, &clDevice, NULL, NULL, &clStatus);
	assert(clStatus == CL_SUCCESS);

	//---- Queue
	clQueue = clCreateCommandQueue(clContext, clDevice, CL_QUEUE_PROFILING_ENABLE, &clStatus);
	assert(clStatus == CL_SUCCESS);
}