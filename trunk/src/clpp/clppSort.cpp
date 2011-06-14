#include "clpp/clppSort.h"

// Push the data on the device
//
// clBuffer_keys    Array of keys for data to be sorted
// datasetSize		Number of elements to be sorted.  Must be <= maxElements passed to the constructor
// keyBits			The number of bits in each key to use for ordering
void clppSort::pushDatas(void* keys, void* values, size_t keySize, size_t valueSize, size_t datasetSize)
{
	_keys = keys;
	_keySize = keySize;

	_values = values;
	_valueSize = valueSize;

	_datasetSize = datasetSize;

	cl_int clStatus;
	_clBuffer_keys = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, keySize * datasetSize, NULL, &clStatus);
	checkCLStatus(clStatus);

	_clBuffer_values = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, valueSize * datasetSize, NULL, &clStatus);
	checkCLStatus(clStatus);

	pushCLDatas(_clBuffer_keys, _clBuffer_values, datasetSize);
}