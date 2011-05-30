#ifndef __CLPP_SCAN_H__
#define __CLPP_SCAN_H__

#include "clpp/clppProgram.h"

class clppScan : public clppProgram
{
public:
	clppScan(clppContext* context);
	~clppScan();

	string getName() { return "Prefix sum (exclusive)"; }

	void scan();

	void pushDatas(void* values, size_t valueSize, size_t datasetSize);
	void pushDatas(cl_mem clBuffer_keys, cl_mem clBuffer_values, size_t datasetSize);

	void popDatas();

private:
	size_t _datasetSize;	// The number of keys to sort

	void* _values;			// The associated data set to sort
	size_t _valueSize;		// The size of a value in bytes

	cl_kernel _kernel_Scan;
	cl_kernel _kernel_UniformAdd;

	cl_mem _clBuffer_values;
	cl_mem _clBuffer_valuesOut;
	cl_mem _clBuffer_Temp;

	size_t _workgroupSize;

	unsigned int* _temp;

	int _blockSumsLevels;
	cl_mem* _clBuffer_BlockSums;
	unsigned int* _blockSumsSizes;

	void allocateBlockSums();
	void freeBlockSums();
};

#endif