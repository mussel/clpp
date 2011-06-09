#ifndef __CLPP_SCAN_GPU_H__
#define __CLPP_SCAN_GPU_H__

#include "clpp/clppScan.h"

class clppScan_GPU : public clppScan
{
public:
	// Create a new scan.
	// maxElements : the maximum number of elements to scan.
	clppScan_GPU(clppContext* context, unsigned int maxElements);
	~clppScan_GPU();

	string getName() { return "Prefix sum (exclusive) for the GPU"; }

	void scan();

	void pushDatas(void* values, void* valuesOut, size_t valueSize, size_t datasetSize);
	void pushDatas(cl_mem clBuffer_keys, cl_mem clBuffer_values, size_t valueSize, size_t datasetSize);

	void popDatas();

	string compilePreprocess(string kernel);

private:
	cl_kernel kernel__scan;
};

#endif