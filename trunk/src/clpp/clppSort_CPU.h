#ifndef __CLPP_SORT_CPU_H__
#define __CLPP_SORT_CPU_H__

#include<math.h>

#include "clpp/clppSort.h"

class clppSort_CPU : public clppSort
{
public:
	clppSort_CPU(clppContext* context, string basePath);

	// Returns the algorithm name
	string getName() { return "CPU Brute force"; }

	// Sort the pushed data set 
	void sort();

	// Push the data on the device
	void pushDatas(void* keys, void* values, size_t valueSize, size_t datasetSize, unsigned int keyBits);

	// Pop the data from the device
	void popDatas();

};

#endif