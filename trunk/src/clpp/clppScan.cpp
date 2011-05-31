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

	_kernel_UniformAdd = clCreateKernel(_clProgram, "kernel__UniformAdd", &clStatus);
	checkCLStatus(clStatus);

	//---- Get the workgroup size
	clGetKernelWorkGroupInfo(_kernel_Scan, _context->clDevice, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_workgroupSize, 0);
	//_workgroupSize = 128;

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
	clStatus  = clSetKernelArg(_kernel_Scan, 2, _workgroupSize * 2 * sizeof(int), 0);
	checkCLStatus(clStatus);
	//_temp = new unsigned int[_workgroupSize * 2];
	//_clBuffer_Temp  = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, _workgroupSize * 2 * sizeof(int), _temp, &clStatus);
	//checkCLStatus(clStatus);
	//clStatus  = clSetKernelArg(_kernel_Scan, 2, sizeof(cl_mem), &_clBuffer_Temp);
	//checkCLStatus(clStatus);

	//---- Apply the scan to each level
    size_t globalWorkSize = {toMultipleOf(_datasetSize, _workgroupSize / 2)};
    size_t localWorkSize = {_workgroupSize / 2};
	cl_mem clValues = _clBuffer_values;
	cl_mem clValuesOut = _clBuffer_valuesOut;

	clStatus = CL_SUCCESS;
	for(unsigned int i = 0; i < _blockSumsLevels; i++)
	{
		clStatus |= clSetKernelArg(_kernel_Scan, 0, sizeof(cl_mem), &clValues);
		clStatus |= clSetKernelArg(_kernel_Scan, 1, sizeof(cl_mem), &clValuesOut);
		clStatus |= clSetKernelArg(_kernel_Scan, 3, sizeof(cl_mem), &_clBuffer_BlockSums[i]);
		clStatus |= clSetKernelArg(_kernel_Scan, 4, sizeof(int), &_blockSumsSizes[i]);
		clStatus |= clEnqueueNDRangeKernel(_context->clQueue, _kernel_Scan, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);

		checkCLStatus(clStatus);

					//clStatus = clFinish(_context->clQueue);     // wait end of read
					//checkCLStatus(clStatus);

		clValues = clValuesOut = _clBuffer_BlockSums[i];
    }

	//----
	for(int i = _blockSumsLevels - 2; i >= 0; i--)
	{
        cl_mem dest = (i > 0) ? _clBuffer_BlockSums[i-1] : _clBuffer_valuesOut;

		clStatus |= clSetKernelArg(_kernel_UniformAdd, 0, sizeof(cl_mem), &dest);
		clStatus |= clSetKernelArg(_kernel_UniformAdd, 1, sizeof(cl_mem), &_clBuffer_BlockSums[i]);
		clStatus |= clSetKernelArg(_kernel_UniformAdd, 2, sizeof(int), &_blockSumsSizes[i]);
		clStatus |= clEnqueueNDRangeKernel(_context->clQueue, _kernel_UniformAdd, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);

		checkCLStatus(clStatus);

			//cl_int clStatus = clFinish(_context->clQueue);     // wait end of read
			//checkCLStatus(clStatus);
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

	//clEnqueueWriteBuffer(_context->clQueue, _clBuffer_values, CL_TRUE, 0, _valueSize * _datasetSize, _values, 0, 0, 0);

	//---- Compute the size of the different block we can use for '_datasetSize' (can be < maxElements)

	// Compute the number of levels requested to do the scan
	_blockSumsLevels = 0;
	unsigned int n = _datasetSize;
	do
	{
		n = (n + _workgroupSize - 1) / _workgroupSize; // round up
		_blockSumsLevels++;
	}
	while(n > 1);

	// Compute the max-size of the blocks
	n = _datasetSize;
	for(unsigned int i = 0; i < _blockSumsLevels; i++)
	{
		_blockSumsSizes[i] = n;
		n = (n + _workgroupSize - 1) / _workgroupSize; // round up
	}
	_blockSumsSizes[_blockSumsLevels] = n;
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

	clStatus = clEnqueueReadBuffer(_context->clQueue, _clBuffer_valuesOut, CL_TRUE, 0, _valueSize * _datasetSize, _values, 0, NULL, NULL);
	checkCLStatus(clStatus);
}

#pragma endregion

#pragma region allocateBlockSums

void clppScan::allocateBlockSums(unsigned int maxElements)
{
	// Compute the number of buffers we need for the scan
	cl_int clStatus;
	_blockSumsLevels = 0;
	unsigned int n = maxElements;
	do
	{
		n = (n + _workgroupSize - 1) / _workgroupSize; // round up
		_blockSumsLevels++;
	}
	while(n > 1);

	// Allocate the arrays
	_clBuffer_BlockSums = new cl_mem[_blockSumsLevels];
	_blockSumsSizes = new unsigned int[_blockSumsLevels + 1];

	// Create the cl-buffers
	n = maxElements;
	for(unsigned int i = 0; i < _blockSumsLevels; i++)
	{
		_blockSumsSizes[i] = n;
		n = (n + _workgroupSize - 1) / _workgroupSize; // round up

		_clBuffer_BlockSums[i] = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, sizeof(int) * n, NULL, &clStatus);
		checkCLStatus(clStatus);
	}
	_blockSumsSizes[_blockSumsLevels] = n;
}

void clppScan::freeBlockSums()
{
	if (!_clBuffer_BlockSums)
		return;

    cl_int clStatus;
    
	for(unsigned int i = 0; i < _blockSumsLevels; i++)
		clStatus = clReleaseMemObject(_clBuffer_BlockSums[i]);

	delete [] _clBuffer_BlockSums;
	delete [] _blockSumsSizes;
	_clBuffer_BlockSums = 0;
	_blockSumsSizes = 0;
}

#pragma endregion