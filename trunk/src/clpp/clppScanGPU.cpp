#include "clpp/clppScanGPU.h"

// Next :
// 1 - Allow templating
// 2 - 

#pragma region Constructor

clppScanGPU::clppScanGPU(clppContext* context, unsigned int maxElements)
{
	_clBuffer_values = 0;
	_clBuffer_valuesOut = 0;
	_clBuffer_BlockSums = 0;

	if (!compile(context, "clppScanGPU.cl"))
		return;

	//---- Prepare all the kernels
	cl_int clStatus;

	kernel__scan = clCreateKernel(_clProgram, "kernel__scan_block_anylength", &clStatus);
	checkCLStatus(clStatus);

	kernel__scanIntra = clCreateKernel(_clProgram, "kernel__scanIntra", &clStatus);
	checkCLStatus(clStatus);

	_kernel_UniformAdd = clCreateKernel(_clProgram, "kernel__UniformAdd", &clStatus);
	checkCLStatus(clStatus);

	//---- Get the workgroup size
	// ATI : Actually the wavefront size is only 64 for the highend cards(48XX, 58XX, 57XX), but 32 for the middleend cards and 16 for the lowend cards.
	// NVidia : 32
	clGetKernelWorkGroupInfo(kernel__scanIntra, _context->clDevice, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_workgroupSize, 0);
	clGetKernelWorkGroupInfo(kernel__scanIntra, _context->clDevice, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &_workgroupSize, 0);
	//_workgroupSize = 128;
	//_workgroupSize = 256;
	//_workgroupSize = 512;

	_workgroupSize = 32;

	//---- Prepare all the buffers
	allocateBlockSums(maxElements);
}

clppScanGPU::~clppScanGPU()
{
	if (_clBuffer_values)
		delete _clBuffer_values;
	if (_clBuffer_valuesOut)
		delete _clBuffer_valuesOut;

	freeBlockSums();
}

#pragma endregion

#pragma region scan

void clppScanGPU::scan()
{
	cl_int clStatus;

	int B = 512;
	//const int nPasses = (float)ceil( ((float)B) / ((float)_workgroupSize) );

	//---- Apply the scan to each level
	size_t globalWorkSize = {toMultipleOf(_datasetSize/B, _workgroupSize)};
	size_t localWorkSize = {_workgroupSize};

	clStatus = clSetKernelArg(kernel__scan, 0, sizeof(cl_mem), &_clBuffer_BlockSums[0]);
	clStatus |= clSetKernelArg(kernel__scan, 1, sizeof(cl_mem), &_clBuffer_values);
	clStatus |= clSetKernelArg(kernel__scan, 2, sizeof(cl_mem), &_clBuffer_valuesOut);
	clStatus |= clSetKernelArg(kernel__scan, 3, sizeof(int), &B);
	clStatus |= clSetKernelArg(kernel__scan, 4, sizeof(int), &_datasetSize);

	clStatus |= clEnqueueNDRangeKernel(_context->clQueue, kernel__scan, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
	checkCLStatus(clStatus);
}

#pragma endregion

#pragma region scan : scan intra + add-uniform

//void clppScanGPU::scan()
//{
//	cl_int clStatus;
//
//	//---- Apply the scan to each level
//	cl_mem clValues = _clBuffer_values;
//	for(unsigned int i = 0; i < _pass; i++)
//	{
//		size_t globalWorkSize = {toMultipleOf(_blockSumsSizes[i], _workgroupSize)};
//		size_t localWorkSize = {_workgroupSize};
//
//		clStatus = clSetKernelArg(kernel__scanIntra, 0, sizeof(cl_mem), &clValues);
//		clStatus |= clSetKernelArg(kernel__scanIntra, 1, sizeof(cl_mem), &_clBuffer_BlockSums[i]);
//		clStatus |= clSetKernelArg(kernel__scanIntra, 2, sizeof(cl_mem), &_blockSumsSizes[i]);
//
//		clStatus |= clEnqueueNDRangeKernel(_context->clQueue, kernel__scanIntra, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
//		checkCLStatus(clStatus);
//
//		// Now we process the sums...
//		clValues = _clBuffer_BlockSums[i];
//    }
//	
//	for(int i = _pass - 2; i >= 0; i--)
//	{
//		size_t globalWorkSize = {toMultipleOf(_blockSumsSizes[i] / 2, _workgroupSize / 2)};
//		size_t localWorkSize = {_workgroupSize / 2};
//
//        cl_mem dest = (i > 0) ? _clBuffer_BlockSums[i-1] : _clBuffer_values;
//
//		clStatus = clSetKernelArg(_kernel_UniformAdd, 0, sizeof(cl_mem), &dest);
//		clStatus |= clSetKernelArg(_kernel_UniformAdd, 1, sizeof(cl_mem), &_clBuffer_BlockSums[i]);
//		clStatus |= clSetKernelArg(_kernel_UniformAdd, 2, sizeof(int), &_blockSumsSizes[i]);
//		checkCLStatus(clStatus);
//
//		clStatus = clEnqueueNDRangeKernel(_context->clQueue, _kernel_UniformAdd, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
//		checkCLStatus(clStatus);
//    }
//}

#pragma endregion

#pragma region pushDatas

void clppScanGPU::pushDatas(void* values, void* valuesOut, size_t valueSize, size_t datasetSize)
{
	//---- Store some values
	_values = values;
	_valuesOut = valuesOut;
	_valueSize = valueSize;
	_datasetSize = datasetSize;

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

	// Compute the block-sum sizes
	n = _datasetSize;
	for(unsigned int i = 0; i < _pass; i++)
	{
		_blockSumsSizes[i] = n;
		n = (n + _workgroupSize - 1) / _workgroupSize; // round up
	}
	_blockSumsSizes[_pass] = n;

	//---- Copy on the device
	cl_int clStatus;
	_clBuffer_values  = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, _valueSize * _datasetSize, _values, &clStatus);
	checkCLStatus(clStatus);
}

void clppScanGPU::pushDatas(cl_mem clBuffer_keys, cl_mem clBuffer_values, size_t datasetSize)
{
}

#pragma endregion

#pragma region popDatas

void clppScanGPU::popDatas()
{
	cl_int clStatus = clEnqueueReadBuffer(_context->clQueue, _clBuffer_valuesOut, CL_TRUE, 0, _valueSize * _datasetSize, _valuesOut, 0, NULL, NULL);
	checkCLStatus(clStatus);
}

#pragma endregion

#pragma region allocateBlockSums

void clppScanGPU::allocateBlockSums(unsigned int maxElements)
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

	_clBuffer_valuesOut = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, sizeof(int) * maxElements, NULL, &clStatus);
	checkCLStatus(clStatus);

	// Create the cl-buffers
	n = maxElements;
	for(unsigned int i = 0; i < _pass; i++)
	{
		_blockSumsSizes[i] = n;

		_clBuffer_BlockSums[i] = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, sizeof(int) * n, NULL, &clStatus);
		checkCLStatus(clStatus);

		n = (n + _workgroupSize - 1) / _workgroupSize; // round up
	}
	_blockSumsSizes[_pass] = n;

	checkCLStatus(clStatus);
}

void clppScanGPU::freeBlockSums()
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