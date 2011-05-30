#include "clpp/clppContext.h"

#include<assert.h>
#include <iostream>

using namespace std;

void clppContext::setup()
{
	setup(0, 0);
}

void clppContext::setup(unsigned int platformId, unsigned int deviceId)
{
	cl_int clStatus;

	//---- Retreive information about platforms
	cl_uint platformsCount;
	clStatus = clGetPlatformIDs(0, NULL, &platformsCount);
	assert(clStatus == CL_SUCCESS);

	cl_platform_id* platforms = new cl_platform_id[platformsCount];
	clStatus = clGetPlatformIDs(platformsCount, platforms, NULL);
	assert(clStatus == CL_SUCCESS);

	platformId = min(platformId, platformsCount - 1);
	clPlatform = platforms[platformId];

	//---- Devices
	cl_uint devicesCount;
	clStatus = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_ALL, 0, NULL, &devicesCount);
	assert(clStatus == CL_SUCCESS);
	assert(devicesCount > 0);
	
	cl_device_id* devices = new cl_device_id[devicesCount];
	clStatus = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_ALL, devicesCount, devices, NULL);
	assert(clStatus == CL_SUCCESS);

	clDevice = devices[min(deviceId, devicesCount - 1)];
	clDevice = devices[0];
	//---- Context
	clContext = clCreateContext(0, 1, &clDevice, NULL, NULL, &clStatus);
	assert(clStatus == CL_SUCCESS);

	//---- Queue
	clQueue = clCreateCommandQueue(clContext, clDevice, CL_QUEUE_PROFILING_ENABLE, &clStatus);
	assert(clStatus == CL_SUCCESS);

	//---- Display some info about the context
	char platformName[500];
	clGetPlatformInfo(clPlatform, CL_PLATFORM_NAME, 500, platformName, NULL);

	cl_device_type deviceType;
	clStatus = clGetDeviceInfo(clDevice, CL_DEVICE_TYPE, sizeof(cl_device_type), (void*)&deviceType,NULL);
	
	char deviceName[500];
	clStatus = clGetDeviceInfo(clDevice, CL_DEVICE_NAME, 500, deviceName, NULL);

	cout << "Platform[" << platformName << "] Device[" << deviceName << "]" << endl;
}
