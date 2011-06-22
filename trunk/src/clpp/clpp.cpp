#include "clpp/clpp.h"

#include "clpp/clppScan_Default.h"
#include "clpp/clppScan_GPU.h"

#include "clpp/clppSort_Blelloch.h"
#include "clpp/clppSort_RadixSort.h"

clppScan* clpp::createBestScan(clppContext* context, size_t valueSize, unsigned int maxElements)
{
	if (context->isGPU)// && context->Vendor == clppVendor::Vendor_NVidia)
		return new clppScan_GPU(context, valueSize, maxElements);

	return new clppScan_Default(context, valueSize, maxElements);
}

clppSort* clpp::createBestSort(clppContext* context, unsigned int maxElements, unsigned int bits)
{
	return new clppSort_Blelloch(context, maxElements);
}

clppSort* clpp::createBestSortKV(clppContext* context, unsigned int maxElements, unsigned int bits)
{
	return new clppSort_RadixSort(context, maxElements, bits, true);
}