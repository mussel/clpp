#ifndef __CLPP_SORT_CPU_H__
#define __CLPP_SORT_CPU_H__

#include "clpp/clppSort.h"

class clppSort_CPU : public clppSort
{
public:
	clppSort_CPU(clppContext* context);

	string getName() { return "CPU Brute force"; }

	void sort();

	void pushDatas(void* keys, void* values, size_t keySize, size_t valueSize, size_t datasetSize, unsigned int keyBits);
	void pushDatas(cl_mem clBuffer_keys, cl_mem clBuffer_values, size_t datasetSize, unsigned int keyBits);

	void popDatas();

	void waitCompletion() {}
};

#endif