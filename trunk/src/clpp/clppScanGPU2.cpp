#include "clpp/clppScanGPU2.h"

#include <iostream>

// Next :
// 1 - Allow templating
// 2 - 

#define SIMT_SIZE 32

#pragma region Constructor

clppScanGPU2::clppScanGPU2(clppContext* context, unsigned int maxElements)
{
	cl_int clStatus;
	_clBuffer_values = 0;
	_clBuffer_valuesOut = 0;

	//---- Compilation
	if (!compile(context, "clppScanGPU2.cl"))
		return;

	//---- Prepare all the kernels
	kernel__scan = clCreateKernel(_clProgram, "kernel__scan_block_anylength", &clStatus);
	checkCLStatus(clStatus);

	//---- Get the workgroup size
	// ATI : Actually the wavefront size is only 64 for the highend cards(48XX, 58XX, 57XX), but 32 for the middleend cards and 16 for the lowend cards.
	// NVidia : 32
	clGetKernelWorkGroupInfo(kernel__scan, _context->clDevice, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &_workgroupSize, 0);
	//clGetKernelWorkGroupInfo(kernel__scan, _context->clDevice, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &_workgroupSize, 0);
}

clppScanGPU2::~clppScanGPU2()
{
	if (_clBuffer_values)
		delete _clBuffer_values;
	if (_clBuffer_valuesOut)
		delete _clBuffer_valuesOut;
}

#pragma endregion

#pragma region compilePreprocess

string clppScanGPU2::compilePreprocess(string kernel)
{
	//ostringstream lines;

	//if (SIMT_SIZE == 32)
	//{
	//	lines << "#define SIMT_SIZE 32" << std::endl;
	//	lines << "#define WORKGROUP_SIZE " <<  _workgroupSize << std::endl;
	//}
	//else
	//{
	//	lines << "#define SIMT_SIZE 64" << std::endl;
	//	lines << "#define WORKGROUP_SIZE " <<  _workgroupSize << std::endl;		
	//}

	//lines << kernel << std::endl;
	//return lines.str();

	return kernel;
}

#pragma endregion

#pragma region scan

void clppScanGPU2::scan()
{
	cl_int clStatus;

	int multiple = _datasetSize - (_datasetSize%_workgroupSize);
	int multipleFactor = multiple / _workgroupSize;

	int B = _workgroupSize * multipleFactor;

	//---- Apply the scan to each level
	size_t globalWorkSize = {toMultipleOf(_datasetSize / multipleFactor, _workgroupSize)};
	//size_t globalWorkSize = {toMultipleOf((_datasetSize+multipleFactor) / multipleFactor, _workgroupSize)};
	size_t localWorkSize = {_workgroupSize};

	const unsigned int nPasses = (float)ceil( B / ((float)_workgroupSize) );

	unsigned int i = 0;
	clStatus  = clSetKernelArg(kernel__scan, i++, _workgroupSize * _valueSize, 0);
	clStatus |= clSetKernelArg(kernel__scan, i++, sizeof(cl_mem), &_clBuffer_values);
	clStatus |= clSetKernelArg(kernel__scan, i++, sizeof(cl_mem), &_clBuffer_valuesOut);
	clStatus |= clSetKernelArg(kernel__scan, i++, sizeof(int), &B);
	clStatus |= clSetKernelArg(kernel__scan, i++, sizeof(int), &_datasetSize);
	clStatus |= clSetKernelArg(kernel__scan, i++, sizeof(int), &nPasses);

	clStatus |= clEnqueueNDRangeKernel(_context->clQueue, kernel__scan, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
	checkCLStatus(clStatus);
}

#pragma endregion

#pragma region pushDatas

void clppScanGPU2::pushDatas(void* values, void* valuesOut, size_t valueSize, size_t datasetSize)
{
	//---- Store some values
	_values = values;
	_valuesOut = valuesOut;
	_valueSize = valueSize;
	_datasetSize = datasetSize;

	//---- Copy on the device
	cl_int clStatus;
	
	_clBuffer_values = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, _valueSize * _datasetSize, _values, &clStatus);
	checkCLStatus(clStatus);

	_clBuffer_valuesOut = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, _valueSize * _datasetSize, _values, &clStatus);
	checkCLStatus(clStatus);
}

void clppScanGPU2::pushDatas(cl_mem clBuffer_keys, cl_mem clBuffer_values, size_t datasetSize)
{
}

#pragma endregion

#pragma region popDatas

void clppScanGPU2::popDatas()
{
	cl_int clStatus = clEnqueueReadBuffer(_context->clQueue, _clBuffer_valuesOut, CL_TRUE, 0, _valueSize * _datasetSize, _valuesOut, 0, NULL, NULL);
	checkCLStatus(clStatus);
}

#pragma endregion