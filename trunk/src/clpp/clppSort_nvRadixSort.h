#ifndef __CLPP_SORT_NVRADIXSORT_H__
#define __CLPP_SORT_NVRADIXSORT_H__

#include<math.h>

#include "clpp/clppSort.h"

class clppSort_nvRadixSort : public clppSort
{
public:
	clppSort_nvRadixSort(clppContext* context, string basePath);

	// Returns the algorithm name
	string getName() { return "NVidia Radix Sort"; }

	// Sort the pushed data set 
	void sort();

	// Sorts input arrays of unsigned integer keys and (optional) values
	void sort(cl_mem clBuffer_keys, unsigned int elementsCount, unsigned int keyBits);

	// Push the data on the device
	//
	// clBuffer_keys    Array of keys for data to be sorted
	// datasetSize		Number of elements to be sorted.  Must be <= maxElements passed to the constructor
	// keyBits			The number of bits in each key to use for ordering
	void pushDatas(cl_mem* clBuffer_keys, cl_mem* values, size_t datasetSize, unsigned int keyBits);

	// Pop the data from the device
	void popDatas();

private:
	cl_kernel kernel_RadixSortBlocksKeysOnly;
	cl_kernel kernel_FindRadixOffsets;
	cl_kernel kernel_ScanNaive;
	cl_kernel kernel_ReorderDataKeysOnly;

	// Main key-only radix sort function.  Sorts in place in the keys and values
	// arrays, but uses the other device arrays as temporary storage.  All pointer
	// parameters are device pointers.  Uses cudppScan() for the prefix sum of
	// radix counters.
	void radixSortKeysOnly(cl_mem d_keys, unsigned int numElements, unsigned int keyBits);

	// Perform one step of the radix sort.  Sorts by nbits key bits per step, starting at startbit.
	void radixSortStepKeysOnly(cl_mem d_keys, unsigned int nbits, unsigned int startbit, unsigned int numElements);
	void radixSortBlocksKeysOnlyOCL(cl_mem d_keys, unsigned int nbits, unsigned int startbit, unsigned int numElements);
	void findRadixOffsetsOCL(unsigned int startbit, unsigned int numElements);
	void scanNaiveOCL(unsigned int numElements);
	void reorderDataKeysOnlyOCL(cl_mem d_keys, unsigned int startbit, unsigned int numElements);
};

#endif