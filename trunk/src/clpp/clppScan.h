#ifndef __CLPP_SCAN_H__
#define __CLPP_SCAN_H__

#include "clpp/clppProgram.h"

class clppScan : public clppProgram
{
public:
	// Create a new scan.
	// maxElements : the maximum number of elements to scan.
	clppScan(clppContext* context, unsigned int maxElements);
	~clppScan();

	string getName() { return "Prefix sum (exclusive)"; }

	void scan();

	void pushDatas(void* values, void* valuesOut, size_t valueSize, size_t datasetSize);
	void pushDatas(cl_mem clBuffer_values, cl_mem clBuffer_valuesOut, size_t valueSize, size_t datasetSize);

	void popDatas();

private:
	size_t _datasetSize;	// The number of keys to sort

	void* _values;			// The associated data set to scan
	void* _valuesOut;			// The scanned data set
	size_t _valueSize;		// The size of a value in bytes

	cl_kernel _kernel_Scan;
	cl_kernel _kernel_ScanSmall;
	cl_kernel _kernel_UniformAdd;

	cl_mem _clBuffer_values;
	cl_mem _clBuffer_valuesOut;
	cl_mem _clBuffer_Temp;

	size_t _workgroupSize;

	unsigned int* _temp;

	int _pass;
	cl_mem* _clBuffer_BlockSums;
	unsigned int* _blockSumsSizes;

	void allocateBlockSums(unsigned int maxElements);
	void freeBlockSums();
};

#endif