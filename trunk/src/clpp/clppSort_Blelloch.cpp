#include "clpp/clppSort_Blelloch.h"

#include<assert.h>

clppSort_Blelloch::clppSort_Blelloch(clppContext* context, string basePath)
{
	//---- Read the source code
	_kernelSource = loadKernelSource(basePath + "clppSort_Blelloch.cl");

	cl_int clStatus;
	const char* ptr = _kernelSource.c_str();
	size_t len = _kernelSource.length();

	//---- Build the program
	clProgram = clCreateProgramWithSource(context->clContext, 1, (const char **)&ptr, &len, &clStatus);
	assert(clStatus == CL_SUCCESS);

	clStatus = clBuildProgram(clProgram, 0, NULL, NULL, NULL, NULL);
  
	if (clStatus != CL_SUCCESS)
	{
		size_t len;
		char buffer[5000];
		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(clProgram, context->clDevice, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);

		//printf("%s\n", buffer);
		printf("%s\n", getOpenCLErrorString(clStatus));

		assert(clStatus == CL_SUCCESS);
	}

	//---- Prepare all the kernels
	kernel_Histogram = clCreateKernel(clProgram, "histogram", &clStatus);
	assert(clStatus == CL_SUCCESS);
	
	kernel_ScanHistogram = clCreateKernel(clProgram, "scanhistograms", &clStatus);
	assert(clStatus == CL_SUCCESS);

	kernel_PasteHistogram = clCreateKernel(clProgram, "pastehistograms", &clStatus);
	assert(clStatus == CL_SUCCESS);

	kernel_Reorder = clCreateKernel(clProgram, "reorder", &clStatus);
	assert(clStatus == CL_SUCCESS);

	kernel_Transpose = clCreateKernel(clProgram, "transpose", &clStatus);
	assert(clStatus == CL_SUCCESS);
}

void clppSort_Blelloch::sort(void* keys, void* values, size_t datasetSize, unsigned int keyBits)
{
	//---- Send the data to the devices

	//----

	//---- Retreive the data from the devices
}