#ifndef __CLPP_SORT_BLELLOCH_H__
#define __CLPP_SORT_BLELLOCH_H__

#include "clpp/clppSort.h"

class clppSort_Blelloch : public clppSort
{
public:
	clppSort_Blelloch(string basePath);

	void sort(clppContext* context, void* keys, void* values, size_t datasetSize, unsigned int keyBits);

private:
	string _kernelSource;
};

#endif