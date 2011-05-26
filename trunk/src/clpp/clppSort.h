#ifndef __CLPP_SORT_H__
#define __CLPP_SORT_H__

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <assert.h>

#include "clpp/clppContext.h"

using namespace std;

// Base class to sort a set of datas with clpp
class clppSort
{
public:
	// Sort the pushed data set 
	virtual void sort() = 0;

	// Push the data on the device
	virtual void pushDatas(void* keys, void* values, size_t valueSize, size_t datasetSize, unsigned int keyBits) = 0;

	// Pop the data from the device
	virtual void popDatas() = 0;

protected:
	// Load the kernel source code
	static string loadKernelSource(string path);
	static const char* getOpenCLErrorString(cl_int err);
	static void checkCLStatus(cl_int clStatus);

	clppContext* _context;

	void* _keys;			// The set of key to sort
	void* _values;			// The associated data set to sort
	size_t _valueSize;		// The size of a value in bytes
	size_t _datasetSize;	// The number of keys to sort
	unsigned int _keyBits;	// The bits used by the key
};

#endif