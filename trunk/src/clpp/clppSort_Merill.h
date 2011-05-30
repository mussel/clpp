// http://code.google.com/p/back40computing/wiki/RadixSorting

#ifndef __CLPP_SORT_MERILL_H__
#define __CLPP_SORT_MERILL_H__

#include "clpp/clppSort.h"

class clppSort_Merill : public clppSort
{
public:
	clppSort_Merill(clppContext* context, unsigned int maxElements);

	string getName() { return "Merill"; }

	void sort();

	void pushDatas(void* keys, void* values, size_t keySize, size_t valueSize, size_t datasetSize, unsigned int keyBits);
	void pushDatas(cl_mem clBuffer_keys, cl_mem clBuffer_values, size_t datasetSize, unsigned int keyBits);

	void popDatas();
};

#endif