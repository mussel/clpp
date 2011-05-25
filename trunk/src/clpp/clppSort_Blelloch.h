#ifndef __CLPP_SORT_BLELLOCH_H__
#define __CLPP_SORT_BLELLOCH_H__

#include "clpp/clppSort.h"

class clppSort_Blelloch : public clppSort
{
public:
	clppSort_Blelloch(clppContext* context, string basePath);

	void sort(void* keys, void* values, size_t datasetSize, unsigned int keyBits);

private:
	string _kernelSource;

	cl_program clProgram;
	cl_kernel kernel_Histogram;
	cl_kernel kernel_ScanHistogram;
	cl_kernel kernel_PasteHistogram;
	cl_kernel kernel_Reorder;
	cl_kernel kernel_Transpose;
};

#endif