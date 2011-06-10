#include "clpp/clpp.h"

#include "clpp/clppScan_Default.h"
#include "clpp/clppScan_GPU.h"

#include "clpp/clppSort_Blelloch.h"

clppScan* clpp::createBestScan(clppContext* context, size_t valueSize, unsigned int maxElements)
{
	if (context->isGPU)
		return new clppScan_GPU(context, valueSize, maxElements);
	return new clppScan_Default(context, valueSize, maxElements);
}

clppSort* clpp::createBestSort(clppContext* context, unsigned int maxElements)
{
	return new clppSort_Blelloch(context, maxElements);
}