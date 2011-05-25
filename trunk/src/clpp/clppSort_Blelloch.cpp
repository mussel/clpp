#include "clpp/clppSort_Blelloch.h"

clppSort_Blelloch::clppSort_Blelloch(string basePath)
{
	_kernelSource = loadKernelSource(basePath + "clppSort_Blelloch.cl");
}

void clppSort_Blelloch::sort(clppContext* context, void* keys, void* values, size_t datasetSize, unsigned int keyBits)
{
	//---- Send the data to the devices

	//---- Retreive the data from the devices
}