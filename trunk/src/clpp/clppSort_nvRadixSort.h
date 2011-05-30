// Four step algorithms from Satish, Harris & Garland

#ifndef __CLPP_SORT_NVRADIXSORT_H__
#define __CLPP_SORT_NVRADIXSORT_H__

#include<math.h>

#include "clpp/clppSort.h"
#include "clpp/clppSort_nvScan.h"

class clppSort_nvRadixSort : public clppSort
{
public:
	clppSort_nvRadixSort(clppContext* context, unsigned int maxElements, const int ctaSize, bool keysOnly = true);
	~clppSort_nvRadixSort();

	string getName() { return "NVidia Radix Sort"; }

	void sort();

	void pushDatas(cl_mem clBuffer_keys, cl_mem clBuffer_values, size_t datasetSize, unsigned int keyBits);

	void popDatas();

private:
	cl_mem _clBuffer_TempKeys;						// Memory objects for original keys and work space
	cl_mem _clBuffer_Counters;						// Counter for each radix
	cl_mem _clBuffer_CountersSum;					// Prefix sum of radix counters
	cl_mem _clBuffer_BlockOffsets;					// Global offsets of each radix in each block

	int CTA_SIZE; // Number of threads per block
    static const unsigned int WARP_SIZE = 32;
	static const unsigned int bitStep = 4;

	unsigned int  mNumElements;     // Number of elements of temp storage allocated
    unsigned int* mTempValues;      // Intermediate storage for values
         
	clppSort_nvScan scan;

	cl_kernel _kernel_RadixSortBlocksKeysOnly;
	cl_kernel _kernel_FindRadixOffsets;
	cl_kernel _kernel_ScanNaive;
	cl_kernel _kernel_ReorderDataKeysOnly;

	// Main key-only radix sort function.  Sorts in place in the keys and values
	// arrays, but uses the other device arrays as temporary storage.  All pointer
	// parameters are device pointers.  Uses cudppScan() for the prefix sum of
	// radix counters.
	void radixSortKeysOnly();

	// Perform one step of the radix sort.  Sorts by nbits key bits per step, starting at startbit.
	void radixSortStepKeysOnly(unsigned int nbits, unsigned int startbit);
	void radixSortBlocksKeysOnlyOCL(unsigned int nbits, unsigned int startbit);
	void findRadixOffsetsOCL(unsigned int startbit);
	//void scanNaiveOCL(unsigned int numElements);
	void reorderDataKeysOnlyOCL(unsigned int startbit);
};

#endif