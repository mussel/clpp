#ifndef __CLPP_SORT_RADIXSORT_H__
#define __CLPP_SORT_RADIXSORT_H__

#include "clpp/clppSort.h"
#include "clpp/clppScan.h"

class clppSort_RadixSort : public clppSort
{
public:
	clppSort_RadixSort(clppContext* context, unsigned int maxElements, unsigned int bits);
	~clppSort_RadixSort();

	string getName() { return "Radix sort"; }

	void sort();

	void pushDatas(void* dataSet, size_t datasetSize);
	void pushCLDatas(cl_mem clBuffer_dataSet, size_t datasetSize);

	void popDatas();

private:
	size_t _datasetSize;	// The number of keys to sort

	void* _dataSetOut;
	cl_mem _clBuffer_dataSetOut;

	cl_kernel _kernel_RadixLocalSort;
	cl_kernel _kernel_RadixPermute;	

	size_t _workgroupSize;

	unsigned int _bits;

	void radixLocal(cl_mem data, cl_mem hist, cl_mem blockHists, int bitOffset, const unsigned int N);
	void radixPermute(cl_mem dataIn, cl_mem dataOut, cl_mem histScan, cl_mem blockHists, int bitOffset, const unsigned int N);
	void freeUpRadixMems();

	clppScan* _scan;

	cl_mem _clBuffer_radixHist1;
	cl_mem _clBuffer_radixHist2;
	cl_mem radixDataB;
};

#endif