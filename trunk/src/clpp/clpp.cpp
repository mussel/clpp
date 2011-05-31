#include "clpp/clpp.h"

#include "clpp/clppScan.h"
#include "clpp/clppSort_Blelloch.h"

clppScan* clpp::createBestScan(clppContext* context, unsigned int maxElements)
{
	return new clppScan(context, maxElements);
}

clppSort* clpp::createBestSort(clppContext* context, unsigned int maxElements)
{
	return new clppSort_Blelloch(context, maxElements);
}

