#ifndef __CLPP_SORT_NVRADIXSORT_H__
#define __CLPP_SORT_NVRADIXSORT_H__

#include<math.h>

#include "clpp/clppSort.h"
#include "clpp/clppSort_nvScan.h"

class clppSort_nvRadixSort : public clppSort
{
public:
	clppSort_nvRadixSort(clppContext* context, string basePath, unsigned int maxElements, const int ctaSize, bool keysOnly = true);
	~clppSort_nvRadixSort();

	string getName() { return "NVidia Radix Sort"; }

	void sort();

	void pushDatas(cl_mem clBuffer_keys, cl_mem clBuffer_values, size_t datasetSize, unsigned int keyBits);

	void popDatas();

private:
	cl_mem d_tempKeys;                   // Memory objects for original keys and work space
	cl_mem mCounters;                    // Counter for each radix
	cl_mem mCountersSum;                 // Prefix sum of radix counters
	cl_mem mBlockOffsets;                // Global offsets of each radix in each block
    cl_kernel ckRadixSortBlocksKeysOnly; // OpenCL kernels
	cl_kernel ckFindRadixOffsets;
	cl_kernel ckScanNaive;
	cl_kernel ckReorderDataKeysOnly;

	int CTA_SIZE; // Number of threads per block
    static const unsigned int WARP_SIZE = 32;
	static const unsigned int bitStep = 4;

	unsigned int  mNumElements;     // Number of elements of temp storage allocated
    unsigned int* mTempValues;      // Intermediate storage for values
         
	clppSort_nvScan scan;

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