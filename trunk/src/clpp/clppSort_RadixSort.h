#ifndef __CLPP_SORT_RADIXSORT_H__
#define __CLPP_SORT_RADIXSORT_H__

#include "clpp/clppSort.h"
#include "clpp/clppScan.h"

class clppSort_RadixSort : public clppSort
{
public:
	clppSort_RadixSort(clppContext* context, unsigned int maxElements);
	~clppSort_RadixSort();

	string getName() { return "Radix sort"; }

	void sort();

	void pushDatas(void* values, void* valuesOut, size_t keySize, size_t valueSize, size_t datasetSize, unsigned int keyBits);
	void pushCLDatas(cl_mem clBuffer_keys, cl_mem clBuffer_values, size_t datasetSize, unsigned int keyBits) {}

	void popDatas();

private:
	size_t _datasetSize;	// The number of keys to sort

	void* _valuesOut;

	cl_kernel _kernel_RadixLocalSort;
	cl_kernel _kernel_RadixPermute;

	cl_mem _clBuffer_values;
	cl_mem _clBuffer_valuesOut;

	size_t _workgroupSize;

	void radixLocal(cl_mem data, cl_mem hist, cl_mem blockHists, int bitOffset, const unsigned int N);
	void radixPermute(cl_mem dataIn, cl_mem dataOut, cl_mem histScan, cl_mem blockHists, int bitOffset, const unsigned int N);
	void freeUpRadixMems();

	clppScan* _scan;

	cl_mem radixHist1;
	cl_mem radixHist2;
	cl_mem radixDataB;
	int radixSortAllocatedForN;
};

#endif