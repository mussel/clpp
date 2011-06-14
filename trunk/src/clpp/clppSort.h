#ifndef __CLPP_SORT_H__
#define __CLPP_SORT_H__

#include "clpp/clppContext.h"
#include "clpp/clppProgram.h"

using namespace std;

// Base class to sort a set of datas with clpp
class clppSort : public clppProgram
{
public:
	// Returns the algorithm name
	virtual string getName() = 0;

	// Sort the pushed data set 
	virtual void sort() = 0;

	// Push the data on the device
	//
	// datasetSize		Number of elements to be sorted.  Must be <= maxElements passed to the constructor
	// keyBits			The number of bits in each key to use for ordering
	virtual void pushDatas(void* keys, void* values, size_t keySize, size_t valueSize, size_t datasetSize, unsigned int keyBits);

	// Push the data on the device
	//
	// clBuffer_keys    Array of keys for data to be sorted
	// datasetSize		Number of elements to be sorted.  Must be <= maxElements passed to the constructor
	// keyBits			The number of bits in each key to use for ordering
	virtual void pushCLDatas(cl_mem clBuffer_keys, cl_mem clBuffer_values, size_t datasetSize, unsigned int keyBits) = 0;

	// Pop the data from the device
	virtual void popDatas() = 0;

protected:
	
	void* _keys;			// The set of key to sort
	cl_mem _clBuffer_keys;	// The cl buffers for the keys
	size_t _keySize;		// The size of a key in bytes

	void* _values;			// The associated data set to sort
	cl_mem _clBuffer_values;// The cl buffers for the values
	size_t _valueSize;		// The size of a value in bytes

	size_t _datasetSize;	// The number of keys to sort

	unsigned int _keyBits;	// The bits used by the key
};

#endif