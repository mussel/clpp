#ifndef __CLPP_SCANGPU_H__
#define __CLPP_SCANGPU_H__

#include "clpp/clppProgram.h"

class clppScanGPU : public clppProgram
{
public:
	// Create a new scan.
	// maxElements : the maximum number of elements to scan.
	clppScanGPU(clppContext* context, unsigned int maxElements);
	~clppScanGPU();

	string getName() { return "Prefix sum (exclusive)"; }

	void scan();

	void pushDatas(void* values, void* valuesOut, size_t valueSize, size_t datasetSize);
	void pushDatas(cl_mem clBuffer_keys, cl_mem clBuffer_values, size_t datasetSize);

	void popDatas();

private:
	size_t _datasetSize;	// The number of keys to sort

	void* _values;			// The associated data set to scan
	void* _valuesOut;		// The scanned data set
	size_t _valueSize;		// The size of a value in bytes

	cl_kernel kernel__scan;

	cl_kernel kernel__scanIntra;
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