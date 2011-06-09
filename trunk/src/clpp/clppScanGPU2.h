#ifndef __CLPP_SCANGPU2_H__
#define __CLPP_SCANGPU2_H__

#include "clpp/clppProgram.h"

class clppScanGPU2 : public clppProgram
{
public:
	// Create a new scan.
	// maxElements : the maximum number of elements to scan.
	clppScanGPU2(clppContext* context, unsigned int maxElements);
	~clppScanGPU2();

	string getName() { return "Prefix sum (exclusive) for the GPU"; }

	void scan();

	void pushDatas(void* values, void* valuesOut, size_t valueSize, size_t datasetSize);
	void pushDatas(cl_mem clBuffer_keys, cl_mem clBuffer_values, size_t datasetSize);

	void popDatas();

	string compilePreprocess(string kernel);

private:
	size_t _datasetSize;	// The number of keys to sort

	void* _values;			// The associated data set to scan
	void* _valuesOut;		// The scanned data set
	size_t _valueSize;		// The size of a value in bytes

	cl_kernel kernel__scan;

	cl_mem _clBuffer_values;
	cl_mem _clBuffer_valuesOut;

	size_t _workgroupSize;
};

#endif