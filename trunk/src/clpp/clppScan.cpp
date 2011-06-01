#include "clpp/clppScan.h"

// Next :
// 1 - Allow templating
// 2 - 

#pragma region Constructor

clppScan::clppScan(clppContext* context, unsigned int maxElements)
{
	_clBuffer_values = 0;
	_clBuffer_valuesOut = 0;
	_clBuffer_BlockSums = 0;

	if (!compile(context, "clppScan.cl"))
		return;

	//---- Prepare all the kernels
	cl_int clStatus;

	_kernel_Scan = clCreateKernel(_clProgram, "kernel__ExclusivePrefixScan", &clStatus);
	checkCLStatus(clStatus);

	_kernel_ScanSmall = clCreateKernel(_clProgram, "kernel__ExclusivePrefixScanSmall", &clStatus);
	checkCLStatus(clStatus);

	_kernel_UniformAdd = clCreateKernel(_clProgram, "kernel__UniformAdd", &clStatus);
	checkCLStatus(clStatus);

	//---- Get the workgroup size
	clGetKernelWorkGroupInfo(_kernel_Scan, _context->clDevice, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_workgroupSize, 0);
	//_workgroupSize = 128;
	//_workgroupSize = 256;

	//---- Prepare all the buffers
	allocateBlockSums(maxElements);
}

clppScan::~clppScan()
{
	if (_clBuffer_values)
		delete _clBuffer_values;
	if (_clBuffer_valuesOut)
		delete _clBuffer_valuesOut;

	freeBlockSums();
}

#pragma endregion

#pragma region scan

size_t toMultipleOf(size_t N, size_t base)
{
	return (ceil((double)N / (double)base) * base);
}

void clppScan::scan()
{
	cl_int clStatus;

	// Intel SDK problem
	//clStatus  = clSetKernelArg(_kernel_Scan, 2, _workgroupSize * 2 * sizeof(int), 0);
	clStatus  = clSetKernelArg(_kernel_Scan, 2, _workgroupSize * sizeof(int), 0);
	checkCLStatus(clStatus);
	
	//---- Apply the scan to each level
    size_t globalWorkSize = {toMultipleOf(_datasetSize, _workgroupSize / 2)};
    size_t localWorkSize = {_workgroupSize / 2};
	cl_mem clValues = _clBuffer_values;
	cl_mem clValuesOut = _clBuffer_valuesOut;

	unsigned int blockSize = localWorkSize / 2;

	clStatus = CL_SUCCESS;
	for(unsigned int i = 0; i < _pass; i++)
	{
		clStatus |= clSetKernelArg(_kernel_Scan, 0, sizeof(cl_mem), &clValues);
		clStatus |= clSetKernelArg(_kernel_Scan, 1, sizeof(cl_mem), &clValuesOut);
		clStatus |= clSetKernelArg(_kernel_Scan, 3, sizeof(cl_mem), &_clBuffer_BlockSums[i]);
		clStatus |= clSetKernelArg(_kernel_Scan, 4, sizeof(int), &_blockSumsSizes[i]);
		clStatus |= clEnqueueNDRangeKernel(_context->clQueue, _kernel_Scan, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);

		checkCLStatus(clStatus);

		clValues = clValuesOut = _clBuffer_BlockSums[i];
    }

	//---- Last pass
	//clStatus |= clSetKernelArg(_kernel_ScanSmall, 0, sizeof(cl_mem), &_clBuffer_BlockSums[_pass - 2]);
	//clStatus |= clSetKernelArg(_kernel_ScanSmall, 1, sizeof(cl_mem), &clValuesOut);
	//clStatus |= clSetKernelArg(_kernel_ScanSmall, 2, _workgroupSize * 2 * sizeof(int), 0);
	////clStatus |= clSetKernelArg(_kernel_ScanSmall, 3, sizeof(cl_mem), &_clBuffer_BlockSums[i]);
	//clStatus |= clSetKernelArg(_kernel_ScanSmall, 3, sizeof(int), &_blockSumsSizes[_pass - 1]);
	//clStatus |= clEnqueueNDRangeKernel(_context->clQueue, _kernel_ScanSmall, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);

	//checkCLStatus(clStatus);

	//---- Uniform addition
	for(int i = _pass - 2; i >= 0; i--)
	{
        cl_mem dest = (i > 0) ? _clBuffer_BlockSums[i-1] : _clBuffer_valuesOut;

		clStatus |= clSetKernelArg(_kernel_UniformAdd, 0, sizeof(cl_mem), &dest);
		clStatus |= clSetKernelArg(_kernel_UniformAdd, 1, sizeof(cl_mem), &_clBuffer_BlockSums[i]);
		clStatus |= clSetKernelArg(_kernel_UniformAdd, 2, sizeof(int), &_blockSumsSizes[i]);
		clStatus |= clEnqueueNDRangeKernel(_context->clQueue, _kernel_UniformAdd, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);

		checkCLStatus(clStatus);
    }
}

#pragma endregion

#pragma region pushDatas

void clppScan::pushDatas(void* values, void* valuesOut, size_t valueSize, size_t datasetSize)
{
	//---- Store some values
	_values = values;
	_valuesOut = valuesOut;
	_valueSize = valueSize;
	_datasetSize = datasetSize;

	//---- Copy on the device
	cl_int clStatus;
	_clBuffer_values  = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, _valueSize * _datasetSize, _values, &clStatus);
	checkCLStatus(clStatus);

	_clBuffer_valuesOut  = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, _valueSize * _datasetSize, _valuesOut, &clStatus);
	checkCLStatus(clStatus);

	//---- Compute the size of the different block we can use for '_datasetSize' (can be < maxElements)
	// Compute the number of levels requested to do the scan
	_pass = 0;
	unsigned int n = _datasetSize;
	do
	{
		n = (n + _workgroupSize - 1) / _workgroupSize; // round up
		_pass++;
	}
	while(n > 1);

	// Compute the blocks sizes
	n = _datasetSize;
	for(unsigned int i = 0; i < _pass; i++)
	{
		_blockSumsSizes[i] = n;
		n = (n + _workgroupSize - 1) / _workgroupSize; // round up
	}
	_blockSumsSizes[_pass] = n;

	//---- V2
	//float t = log((float)_datasetSize) / log((float)_workgroupSize/2);
	//_pass = (cl_uint)t;

 //   // If t is equal to pass
 //   if(fabs(t - (float)_pass) >= 1e-7)
 //       _pass--;

	//// Compute the max-size of each blocks
	//for(unsigned int pass = 0; pass < _pass; pass++)
	//	_blockSumsSizes[pass] = (int)(_datasetSize / pow((float)_workgroupSize, (float)pass));
	//_blockSumsSizes[_pass] = 1;
}

void clppScan::pushDatas(cl_mem clBuffer_keys, cl_mem clBuffer_values, size_t datasetSize)
{
}

#pragma endregion

#pragma region popDatas

void clppScan::popDatas()
{
    cl_int clStatus = clFinish(_context->clQueue);     // wait end of read
	checkCLStatus(clStatus);

	clStatus = clEnqueueReadBuffer(_context->clQueue, _clBuffer_valuesOut, CL_TRUE, 0, _valueSize * _datasetSize, _valuesOut, 0, NULL, NULL);
	checkCLStatus(clStatus);
}

#pragma endregion

#pragma region allocateBlockSums

void clppScan::allocateBlockSums(unsigned int maxElements)
{
	// Compute the number of buffers we need for the scan
	cl_int clStatus;
	_pass = 0;
	unsigned int n = maxElements;
	do
	{
		n = (n + _workgroupSize - 1) / _workgroupSize; // round up
		_pass++;
	}
	while(n > 1);

	// Allocate the arrays
	_clBuffer_BlockSums = new cl_mem[_pass];
	_blockSumsSizes = new unsigned int[_pass + 1];

	// Create the cl-buffers
	n = maxElements;
	for(unsigned int i = 0; i < _pass; i++)
	{
		_blockSumsSizes[i] = n;
		n = (n + _workgroupSize - 1) / _workgroupSize; // round up

		_clBuffer_BlockSums[i] = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, sizeof(int) * n, NULL, &clStatus);
		checkCLStatus(clStatus);
	}
	_blockSumsSizes[_pass] = n;
}

void clppScan::freeBlockSums()
{
	if (!_clBuffer_BlockSums)
		return;

    cl_int clStatus;
    
	for(unsigned int i = 0; i < _pass; i++)
		clStatus = clReleaseMemObject(_clBuffer_BlockSums[i]);

	delete [] _clBuffer_BlockSums;
	delete [] _blockSumsSizes;
	_clBuffer_BlockSums = 0;
	_blockSumsSizes = 0;
}

#pragma endregion