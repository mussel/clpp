#include "clpp/clppSort_RadixSort.h"
#include "clpp/clpp.h"

// Next :
// 1 - Allow templating
// 2 - 

#pragma region Constructor

clppSort_RadixSort::clppSort_RadixSort(clppContext* context, unsigned int maxElements)
{
	_clBuffer_values = 0;
	_clBuffer_valuesOut = 0;

	if (!compile(context, "clppSort_RadixSort.cl"))
		return;

	//---- Prepare all the kernels
	cl_int clStatus;

	_kernel_RadixLocalSort = clCreateKernel(_clProgram, "kernel__radixLocalSort", &clStatus);
	checkCLStatus(clStatus);

	_kernel_RadixPermute = clCreateKernel(_clProgram, "kernel__radixPermute", &clStatus);
	checkCLStatus(clStatus);

	//---- Get the workgroup size
	clGetKernelWorkGroupInfo(_kernel_RadixLocalSort, _context->clDevice, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_workgroupSize, 0);

	_workgroupSize = 32;

	_scan = clpp::createBestScan(context, maxElements);

	radixSortAllocatedForN = 0;
    radixHist1 = NULL;
    radixHist2 = NULL;
    radixDataB = NULL;
}

clppSort_RadixSort::~clppSort_RadixSort()
{
	if (_clBuffer_values)
		delete _clBuffer_values;
}

#pragma endregion

#pragma region sort

inline int roundUpDiv(int A, int B) { return (A + B - 1) / (B); }

void clppSort_RadixSort::sort()
{
	cl_int clStatus;
    unsigned int numBlocks = roundUpDiv(_datasetSize, _workgroupSize * 4);
    if (radixSortAllocatedForN < _datasetSize)
	{
        freeUpRadixMems();

        radixDataB = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, sizeof(cl_int) * 2 * _datasetSize, NULL, &clStatus);
        checkCLStatus(clStatus);
        radixHist1 = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, sizeof(cl_uint) * 16 * numBlocks, NULL, &clStatus);
        checkCLStatus(clStatus);
        radixHist2 = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, sizeof(cl_uint) * 2 * 16 * numBlocks, NULL, &clStatus);
        checkCLStatus(clStatus);
        radixSortAllocatedForN = _datasetSize;
    }

	cl_mem dataA = _clBuffer_values;
    cl_mem dataB = radixDataB;
    for(unsigned int bitOffset = 0; bitOffset < 32; bitOffset += 4)
	{
        radixLocal(dataA, radixHist1, radixHist2, bitOffset, _datasetSize);
		_scan->pushDatas(radixHist1, radixHist1, _valueSize, 16 * numBlocks);
		_scan->scan();
        radixPermute(dataA, dataB, radixHist1, radixHist2, bitOffset, _datasetSize);
        std::swap(dataA, dataB);
    }
}

void clppSort_RadixSort::freeUpRadixMems()
{
    cl_int err;
    if (radixHist1)
	{
        err = clReleaseMemObject(radixHist1);
        checkCLStatus(err);
        radixHist1 = NULL;
    }
    if (radixHist2)
	{
        err = clReleaseMemObject(radixHist2);
        checkCLStatus(err);
        radixHist2 = NULL;
    }
    if (radixDataB)
	{
        err = clReleaseMemObject(radixDataB);
        checkCLStatus(err);
        radixDataB = NULL;
    }
    radixSortAllocatedForN = 0;    
}

void clppSort_RadixSort::radixLocal(cl_mem data, cl_mem hist, cl_mem blockHists, int bitOffset, const unsigned int N)
{
    int LTYPE_SIZE = sizeof(cl_int);
    //if (extensions->contains("cl_khr_byte_addressable_store"))
    //    LTYPE_SIZE = sizeof(cl_int);
    cl_int clStatus = CL_SUCCESS;
    unsigned int a = 0;
    unsigned int Ndiv4 = roundUpDiv(N, 4);

	clStatus |= clSetKernelArg(_kernel_RadixLocalSort, a++, sizeof(cl_int2) * 4 * _workgroupSize, (const void*)NULL);  // shared,   4*4 int2s
    clStatus |= clSetKernelArg(_kernel_RadixLocalSort, a++, LTYPE_SIZE * 4 * 2 * _workgroupSize, (const void*)NULL);    // indices,  4*4*2 shorts
    clStatus |= clSetKernelArg(_kernel_RadixLocalSort, a++, LTYPE_SIZE * 4 * _workgroupSize, (const void*)NULL);        // sharedSum, 4*4 shorts
    clStatus |= clSetKernelArg(_kernel_RadixLocalSort, a++, sizeof(cl_mem), (const void*)&data);
    clStatus |= clSetKernelArg(_kernel_RadixLocalSort, a++, sizeof(cl_mem), (const void*)&hist);
    clStatus |= clSetKernelArg(_kernel_RadixLocalSort, a++, sizeof(cl_mem), (const void*)&blockHists);
    clStatus |= clSetKernelArg(_kernel_RadixLocalSort, a++, sizeof(int), (const void*)&bitOffset);
    clStatus |= clSetKernelArg(_kernel_RadixLocalSort, a++, sizeof(unsigned int), (const void*)&N);
    checkCLStatus(clStatus);

	size_t global[1] = {toMultipleOf(Ndiv4, _workgroupSize)};
    size_t local[1] = {_workgroupSize};
	clStatus = clEnqueueNDRangeKernel(_context->clQueue, _kernel_RadixLocalSort, 1, NULL, global, local, 0, NULL, NULL);

    clStatus |= clFinish(_context->clQueue);
    checkCLStatus(clStatus);
}

void clppSort_RadixSort::radixPermute(cl_mem dataIn, cl_mem dataOut, cl_mem histScan, cl_mem blockHists, int bitOffset, const unsigned int N)
{
    cl_int clStatus = CL_SUCCESS;
    unsigned int a = 0;
    unsigned int Ndiv4 = roundUpDiv(N, 4);

    clStatus |= clSetKernelArg(_kernel_RadixPermute, a++, sizeof(cl_mem), (const void*)&dataIn);
    clStatus |= clSetKernelArg(_kernel_RadixPermute, a++, sizeof(cl_mem), (const void*)&dataOut);
    clStatus |= clSetKernelArg(_kernel_RadixPermute, a++, sizeof(cl_mem), (const void*)&histScan);
    clStatus |= clSetKernelArg(_kernel_RadixPermute, a++, sizeof(cl_mem), (const void*)&blockHists);
    clStatus |= clSetKernelArg(_kernel_RadixPermute, a++, sizeof(int), (const void*)&bitOffset);
    clStatus |= clSetKernelArg(_kernel_RadixPermute, a++, sizeof(unsigned int), (const void*)&N);
    checkCLStatus(clStatus);
    
	size_t global[1] = {toMultipleOf(Ndiv4, _workgroupSize)};
    size_t local[1] = {_workgroupSize};
    clStatus = clEnqueueNDRangeKernel(_context->clQueue, _kernel_RadixPermute, 1, NULL, global, local, 0, NULL, NULL);
    clStatus |= clFinish(_context->clQueue);

    checkCLStatus(clStatus);
}

#pragma endregion

#pragma region pushDatas

void clppSort_RadixSort::pushDatas(void* values, void* valuesOut, size_t keySize, size_t valueSize, size_t datasetSize, unsigned int keyBits)
{
	//---- Store some values
	_values = values;
	_valuesOut = valuesOut;
	_valueSize = valueSize;
	_keySize = keySize;
	_datasetSize = datasetSize;

	//---- Copy on the device
	cl_int clStatus;
	_clBuffer_values = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (_valueSize+_keySize) * _datasetSize, _values, &clStatus);
	checkCLStatus(clStatus);
	_clBuffer_valuesOut = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (_valueSize+_keySize) * _datasetSize, _values, &clStatus);
	checkCLStatus(clStatus);
}

#pragma endregion

#pragma region popDatas

void clppSort_RadixSort::popDatas()
{
	cl_int clStatus = clEnqueueReadBuffer(_context->clQueue, _clBuffer_valuesOut, CL_TRUE, 0, (_valueSize+_keySize) * _datasetSize, _valuesOut, 0, NULL, NULL);
	checkCLStatus(clStatus);
}

#pragma endregion