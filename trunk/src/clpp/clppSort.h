#ifndef __CLPP_SORT_H__
#define __CLPP_SORT_H__

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>

using namespace std;

// Base class to sort a set of datas with clpp
class clppSort
{
public:
	void sort(void* keys, void* values, size_t datasetSize, unsigned int keyBits);

protected:
	// Load the kernel source code
	static string loadKernelSource(string path);
};

#endif