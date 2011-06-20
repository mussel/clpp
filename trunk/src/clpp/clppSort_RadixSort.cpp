#include "clpp/clppSort_RadixSort.h"
#include "clpp/clpp.h"

#include "clpp/clppScan_Default.h"

// Next :
// 1 - Allow templating
// 2 - Allow to sort on specific bits only
// 3 - Allow sorting of keys only

#pragma region Constructor

clppSort_RadixSort::clppSort_RadixSort(clppContext* context, unsigned int maxElements, unsigned int bits)
{
	_valueSize = 4;
	_keySize = 4;
	_clBuffer_dataSet = 0;
	_clBuffer_dataSetOut = 0;

	_bits = bits;

	if (!compile(context, "clppSort_RadixSort.cl"))
		return;

	//---- Prepare all the kernels
	cl_int clStatus;

	_kernel_RadixLocalSort = clCreateKernel(_clProgram, "kernel__radixLocalSort", &clStatus);
	checkCLStatus(clStatus);

	_kernel_RadixPermute = clCreateKernel(_clProgram, "kernel__radixPermute", &clStatus);
	checkCLStatus(clStatus);

	//---- Get the workgroup size
	//clGetKernelWorkGroupInfo(_kernel_RadixLocalSort, _context->clDevice, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_workgroupSize, 0);
	_workgroupSize = 32;

	_scan = clpp::createBestScan(context, sizeof(int), maxElements);

    _clBuffer_radixHist1 = NULL;
    _clBuffer_radixHist2 = NULL;
	_datasetSize = 0;
	_is_clBuffersOwner = false;
}

clppSort_RadixSort::~clppSort_RadixSort()
{
	if (_is_clBuffersOwner)
	{
		if (_clBuffer_dataSet)
			clReleaseMemObject(_clBuffer_dataSet);

		if (_clBuffer_dataSetOut)
			clReleaseMemObject(_clBuffer_dataSetOut);
	}

	if (_clBuffer_radixHist1)
		clReleaseMemObject(_clBuffer_radixHist1);

	if (_clBuffer_radixHist2)
		clReleaseMemObject(_clBuffer_radixHist2);

	delete _scan;
}

#pragma endregion

string clppSort_RadixSort::compilePreprocess(string kernel)
{
	string source;

	//if (_templateType == Int)
	{
		switch(_bits)
		{
		case 32:
			source = "#define MAX_INT2 (int2)(0x7FFFFFFF,0)";
			break;
		case 28:
			source = "#define MAX_INT2 (int2)(0x7FFFFFF,0)";
			break;
		case 24:
			source = "#define MAX_INT2 (int2)(0x7FFFFF,0)";
			break;
		case 20:
			source = "#define MAX_INT2 (int2)(0x7FFFF,0)";
			break;
		case 16:
			source = "#define MAX_INT2 (int2)(0x7FFF,0)";
			break;
		case 12:
			source = "#define MAX_INT2 (int2)(0x7FF,0)";
			break;
		case 8:
			source = "#define MAX_INT2 (int2)(0x7F,0)";
			break;
		case 4:
			source = "#define MAX_INT2 (int2)(0x7,0)";
			break;
		}

		source += "\n#define K_TYPE int\n";
	}
	/*else if (_templateType == UInt)
	{
		switch(_bits)
		{
		case 32:
			source = "#define MAX_INT2 (int2)0xFFFFFFFF";
			break;
		case 28:
			source = "#define MAX_INT2 (int2)0xFFFFFFF";
			break;
		case 24:
			source = "#define MAX_INT2 (int2)0xFFFFFF";
			break;
		case 20:
			source = "#define MAX_INT2 (int2)0xFFFFF";
			break;
		case 16:
			source = "#define MAX_INT2 (int2)0xFFFF";
			break;
		case 12:
			source = "#define MAX_INT2 (int2)0xFFF";
			break;
		case 8:
			source = "#define MAX_INT2 (int2)0xFF";
			break;
		case 4:
			source = "#define MAX_INT2 (int2)0xF";
			break;
		}

		source += "\n#define K_TYPE uint\n";
	}*/

	return clppSort::compilePreprocess(source + kernel);
}

#pragma region sort

inline int roundUpDiv(int A, int B) { return (A + B - 1) / (B); }

void clppSort_RadixSort::sort()
{
	// Satish et al. empirically set b = 4. The size of a work-group is in hundreds of
	// work-items, depending on the concrete device and each work-item processes more than one
	// stream element, usually 4, in order to hide latencies.

	cl_int clStatus;
    unsigned int numBlocks = roundUpDiv(_datasetSize, _workgroupSize * 4);

	cl_mem dataA = _clBuffer_dataSet;
    cl_mem dataB = _clBuffer_dataSetOut;
    for(unsigned int bitOffset = 0; bitOffset < _bits; bitOffset += 4)
	{
		// 1) Each workgroup sorts its tile by using local memory
		// 2) Create an histogram of d=2^b digits entries
        radixLocal(dataA, _clBuffer_radixHist1, _clBuffer_radixHist2, bitOffset, _datasetSize);
		
		// 3) Scan the p*2^b = p*(16) entry histogram table. Stored in column-major order, computes global digit offsets.
		_scan->pushDatas(_clBuffer_radixHist1, 16 * numBlocks);
		_scan->scan();
        
		// 4) Prefix sum results are used to scatter each work-group's elements to their correct position.
		radixPermute(dataA, dataB, _clBuffer_radixHist1, _clBuffer_radixHist2, bitOffset, _datasetSize);

        std::swap(dataA, dataB);

		//clStatus = clFinish(_context->clQueue);
		//checkCLStatus(clStatus);
    }
}

void clppSort_RadixSort::radixLocal(cl_mem data, cl_mem hist, cl_mem blockHists, int bitOffset, const unsigned int N)
{
    int LTYPE_SIZE = sizeof(cl_int);
    //if (extensions->contains("cl_khr_byte_addressable_store"))
    //    LTYPE_SIZE = sizeof(cl_int);
    cl_int clStatus;
    unsigned int a = 0;
    unsigned int Ndiv4 = roundUpDiv(N, 4);

	clStatus  = clSetKernelArg(_kernel_RadixLocalSort, a++, (_valueSize+_keySize) * 4 * _workgroupSize, (const void*)NULL);	// shared,    4*4 int2s
    clStatus |= clSetKernelArg(_kernel_RadixLocalSort, a++, LTYPE_SIZE * 4 * 2 * _workgroupSize, (const void*)NULL);		// indices,   4*4*2 shorts
    //clStatus |= clSetKernelArg(_kernel_RadixLocalSort, a++, LTYPE_SIZE * 4 * _workgroupSize, (const void*)NULL);			// sharedSum, 4*4 shorts
    clStatus |= clSetKernelArg(_kernel_RadixLocalSort, a++, sizeof(cl_mem), (const void*)&data);
    clStatus |= clSetKernelArg(_kernel_RadixLocalSort, a++, sizeof(cl_mem), (const void*)&hist);
    clStatus |= clSetKernelArg(_kernel_RadixLocalSort, a++, sizeof(cl_mem), (const void*)&blockHists);
    clStatus |= clSetKernelArg(_kernel_RadixLocalSort, a++, sizeof(int), (const void*)&bitOffset);
    clStatus |= clSetKernelArg(_kernel_RadixLocalSort, a++, sizeof(unsigned int), (const void*)&N);
    checkCLStatus(clStatus);

	size_t global[1] = {toMultipleOf(Ndiv4, _workgroupSize)};
    size_t local[1] = {_workgroupSize};
	clStatus = clEnqueueNDRangeKernel(_context->clQueue, _kernel_RadixLocalSort, 1, NULL, global, local, 0, NULL, NULL);

    //clStatus |= clFinish(_context->clQueue);
    //checkCLStatus(clStatus);
}

void clppSort_RadixSort::radixPermute(cl_mem dataIn, cl_mem dataOut, cl_mem histScan, cl_mem blockHists, int bitOffset, const unsigned int N)
{
    cl_int clStatus;
    unsigned int a = 0;
    unsigned int Ndiv4 = roundUpDiv(N, 4);

    clStatus  = clSetKernelArg(_kernel_RadixPermute, a++, sizeof(cl_mem), (const void*)&dataIn);
    clStatus |= clSetKernelArg(_kernel_RadixPermute, a++, sizeof(cl_mem), (const void*)&dataOut);
    clStatus |= clSetKernelArg(_kernel_RadixPermute, a++, sizeof(cl_mem), (const void*)&histScan);
    clStatus |= clSetKernelArg(_kernel_RadixPermute, a++, sizeof(cl_mem), (const void*)&blockHists);
    clStatus |= clSetKernelArg(_kernel_RadixPermute, a++, sizeof(int), (const void*)&bitOffset);
    clStatus |= clSetKernelArg(_kernel_RadixPermute, a++, sizeof(unsigned int), (const void*)&N);
    checkCLStatus(clStatus);
    
	size_t global[1] = {toMultipleOf(Ndiv4, _workgroupSize)};
    size_t local[1] = {_workgroupSize};
    clStatus = clEnqueueNDRangeKernel(_context->clQueue, _kernel_RadixPermute, 1, NULL, global, local, 0, NULL, NULL);

    //clStatus |= clFinish(_context->clQueue);
    //checkCLStatus(clStatus);
}

#pragma endregion

#pragma region pushDatas

void clppSort_RadixSort::pushDatas(void* dataSet, size_t datasetSize)
{
	cl_int clStatus;

	//---- Store some values
	_dataSet = dataSet;
	_dataSetOut = dataSet;
	bool reallocate = datasetSize > _datasetSize || !_is_clBuffersOwner;
	_datasetSize = datasetSize;

	//---- Prepare some buffers
	if (reallocate)
	{
		//---- Release
		if (_clBuffer_dataSet)
		{
			clReleaseMemObject(_clBuffer_dataSet);
			clReleaseMemObject(_clBuffer_dataSetOut);
			clReleaseMemObject(_clBuffer_radixHist1);
			clReleaseMemObject(_clBuffer_radixHist2);
		}

		//---- Allocate
		unsigned int numBlocks = roundUpDiv(_datasetSize, _workgroupSize * 4);
	    
		_clBuffer_radixHist1 = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, _keySize * 16 * numBlocks, NULL, &clStatus);
		checkCLStatus(clStatus);
		_clBuffer_radixHist2 = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, (_valueSize+_keySize) * 16 * numBlocks, NULL, &clStatus);
		checkCLStatus(clStatus);

		//---- Copy on the device
		_clBuffer_dataSet = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, (_valueSize+_keySize) * _datasetSize, _dataSet, &clStatus);
		checkCLStatus(clStatus);

		_clBuffer_dataSetOut = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, (_valueSize+_keySize) * _datasetSize, NULL, &clStatus);
		checkCLStatus(clStatus);

		_is_clBuffersOwner = true;
	}
	else
		// Just resend
		clEnqueueWriteBuffer(_context->clQueue, _clBuffer_dataSet, CL_FALSE, 0, (_valueSize+_keySize) * _datasetSize, _dataSet, 0, 0, 0);
}

void clppSort_RadixSort::pushCLDatas(cl_mem clBuffer_dataSet, size_t datasetSize)
{
	cl_int clStatus;

	_is_clBuffersOwner = false;

	//---- Store some values
	bool reallocate = datasetSize > _datasetSize;
	_datasetSize = datasetSize;

	//---- Prepare some buffers
	if (reallocate)
	{
		//---- Release
		if (_clBuffer_dataSet)
		{
			clReleaseMemObject(_clBuffer_dataSet);
			clReleaseMemObject(_clBuffer_dataSetOut);
			clReleaseMemObject(_clBuffer_radixHist1);
			clReleaseMemObject(_clBuffer_radixHist2);
		}

		//---- Allocate
		unsigned int numBlocks = roundUpDiv(_datasetSize, _workgroupSize * 4);
	    
		// column size = 2^b = 16
		// row size = numblocks
		_clBuffer_radixHist1 = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, _keySize * 16 * numBlocks, NULL, &clStatus);
		checkCLStatus(clStatus);
		_clBuffer_radixHist2 = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, (_valueSize + _keySize) * 16 * numBlocks, NULL, &clStatus);
		checkCLStatus(clStatus);
	}

	_clBuffer_dataSet = clBuffer_dataSet;
	_clBuffer_dataSetOut = clBuffer_dataSet;
}

#pragma endregion

#pragma region popDatas

void clppSort_RadixSort::popDatas()
{
	cl_int clStatus = clEnqueueReadBuffer(_context->clQueue, _clBuffer_dataSetOut, CL_TRUE, 0, (_valueSize + _keySize) * _datasetSize, _dataSetOut, 0, NULL, NULL);
	checkCLStatus(clStatus);
}

#pragma endregion