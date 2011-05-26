/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include "clpp/clppSort_CPU.h"

#include <algorithm>

#include "clppSort_nvRadixSort.h"

extern double time1, time2, time3, time4;

clppSort_nvRadixSort::clppSort_nvRadixSort(clppContext* context, string basePath, unsigned int maxElements, const int ctaSize, bool keysOnly) :
        mNumElements(0),
        mTempValues(0),
        _clBuffer_Counters(0),
        _clBuffer_CountersSum(0),
        _clBuffer_BlockOffsets(0),
        CTA_SIZE(ctaSize),
        scan(context, basePath, maxElements/2/CTA_SIZE*16)
{
	cl_int clStatus;

    unsigned int numBlocks = ((maxElements % (CTA_SIZE * 4)) == 0) ?
                             (maxElements / (CTA_SIZE * 4)) : (maxElements / (CTA_SIZE * 4) + 1);

	_clBuffer_TempKeys = clCreateBuffer(context->clContext, CL_MEM_READ_WRITE, sizeof(unsigned int) * maxElements, NULL, &clStatus);
	_clBuffer_Counters = clCreateBuffer(context->clContext, CL_MEM_READ_WRITE, WARP_SIZE * numBlocks * sizeof(unsigned int), NULL, &clStatus);
	_clBuffer_CountersSum = clCreateBuffer(context->clContext, CL_MEM_READ_WRITE, WARP_SIZE * numBlocks * sizeof(unsigned int), NULL, &clStatus);
	_clBuffer_BlockOffsets = clCreateBuffer(context->clContext, CL_MEM_READ_WRITE, WARP_SIZE * numBlocks * sizeof(unsigned int), NULL, &clStatus); 

	if (!compile(context, basePath, "clppSort_nvRadixSort.cl"))
		return;

	//---- Prepare all the kernels
	_kernel_RadixSortBlocksKeysOnly = clCreateKernel(_clProgram, "radixSortBlocksKeysOnly", &clStatus);
    checkCLStatus(clStatus);

    _kernel_FindRadixOffsets = clCreateKernel(_clProgram, "findRadixOffsets", &clStatus);
    checkCLStatus(clStatus);

    _kernel_ScanNaive = clCreateKernel(_clProgram, "scanNaive", &clStatus);
    checkCLStatus(clStatus);

    _kernel_ReorderDataKeysOnly = clCreateKernel(_clProgram, "reorderDataKeysOnly", &clStatus);
    checkCLStatus(clStatus);
}

clppSort_nvRadixSort::~clppSort_nvRadixSort()
{
    clReleaseKernel(_kernel_RadixSortBlocksKeysOnly);
    clReleaseKernel(_kernel_FindRadixOffsets);
    clReleaseKernel(_kernel_ScanNaive);
    clReleaseKernel(_kernel_ReorderDataKeysOnly);
    
    clReleaseMemObject(_clBuffer_TempKeys);
    clReleaseMemObject(_clBuffer_Counters);
    clReleaseMemObject(_clBuffer_CountersSum);
    clReleaseMemObject(_clBuffer_BlockOffsets);
}

void clppSort_nvRadixSort::pushDatas(cl_mem clBuffer_keys, cl_mem clBuffer_values, size_t datasetSize, unsigned int keyBits)
{
	cl_int clStatus;

	_clBuffer_keys = clBuffer_keys;
	_clBuffer_values = clBuffer_values;

	if (_keys != 0)
	{
		clStatus = clEnqueueWriteBuffer(_context->clQueue, _clBuffer_keys, CL_FALSE, 0, _keySize * _datasetSize, _keys, 0, NULL, NULL);
		checkCLStatus(clStatus);
	}
}

void clppSort_nvRadixSort::popDatas()
{
	cl_int clStatus;

	clStatus = clEnqueueReadBuffer(_context->clQueue, _clBuffer_keys, CL_FALSE, 0, _keySize * _datasetSize, _keys, 0, NULL, NULL);
	checkCLStatus(clStatus);

	clFinish(_context->clQueue);     // wait end of read
}

void clppSort_nvRadixSort::sort()
{
	radixSortKeysOnly();
}

void clppSort_nvRadixSort::radixSortKeysOnly()
{
    int i = 0;
    while (_keyBits > i*bitStep)
    {
        radixSortStepKeysOnly(bitStep, i*bitStep);
        i++;
    }
}

void clppSort_nvRadixSort::radixSortStepKeysOnly(unsigned int nbits, unsigned int startbit)
{
    radixSortBlocksKeysOnlyOCL(nbits, startbit);

    findRadixOffsetsOCL(startbit);

    scan.scanExclusiveLarge(_clBuffer_CountersSum, _clBuffer_Counters, 1, _datasetSize / 2 / CTA_SIZE*16);

    reorderDataKeysOnlyOCL(startbit);
}

void clppSort_nvRadixSort::radixSortBlocksKeysOnlyOCL(unsigned int nbits, unsigned int startbit)
{
    unsigned int totalBlocks = _datasetSize/4/CTA_SIZE;
    size_t globalWorkSize[1] = {CTA_SIZE*totalBlocks};
    size_t localWorkSize[1] = {CTA_SIZE};

    cl_int clStatus;
    clStatus  = clSetKernelArg(_kernel_RadixSortBlocksKeysOnly, 0, sizeof(cl_mem), (void*)&_clBuffer_keys);
    clStatus |= clSetKernelArg(_kernel_RadixSortBlocksKeysOnly, 1, sizeof(cl_mem), (void*)&_clBuffer_TempKeys);
    clStatus |= clSetKernelArg(_kernel_RadixSortBlocksKeysOnly, 2, sizeof(unsigned int), (void*)&nbits);
    clStatus |= clSetKernelArg(_kernel_RadixSortBlocksKeysOnly, 3, sizeof(unsigned int), (void*)&startbit);
    clStatus |= clSetKernelArg(_kernel_RadixSortBlocksKeysOnly, 4, sizeof(unsigned int), (void*)&_datasetSize);
    clStatus |= clSetKernelArg(_kernel_RadixSortBlocksKeysOnly, 5, sizeof(unsigned int), (void*)&totalBlocks);
    clStatus |= clSetKernelArg(_kernel_RadixSortBlocksKeysOnly, 6, 4*CTA_SIZE*sizeof(unsigned int), NULL);
    clStatus |= clEnqueueNDRangeKernel(_context->clQueue, _kernel_RadixSortBlocksKeysOnly, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

    checkCLStatus(clStatus);
}

void clppSort_nvRadixSort::findRadixOffsetsOCL(unsigned int startbit)
{
    unsigned int totalBlocks = _datasetSize/2/CTA_SIZE;
    size_t globalWorkSize[1] = {CTA_SIZE*totalBlocks};
    size_t localWorkSize[1] = {CTA_SIZE};
    
	cl_int clStatus;
    clStatus  = clSetKernelArg(_kernel_FindRadixOffsets, 0, sizeof(cl_mem), (void*)&_clBuffer_TempKeys);
    clStatus |= clSetKernelArg(_kernel_FindRadixOffsets, 1, sizeof(cl_mem), (void*)&_clBuffer_Counters);
    clStatus |= clSetKernelArg(_kernel_FindRadixOffsets, 2, sizeof(cl_mem), (void*)&_clBuffer_BlockOffsets);
    clStatus |= clSetKernelArg(_kernel_FindRadixOffsets, 3, sizeof(unsigned int), (void*)&startbit);
    clStatus |= clSetKernelArg(_kernel_FindRadixOffsets, 4, sizeof(unsigned int), (void*)&_datasetSize);
    clStatus |= clSetKernelArg(_kernel_FindRadixOffsets, 5, sizeof(unsigned int), (void*)&totalBlocks);
    clStatus |= clSetKernelArg(_kernel_FindRadixOffsets, 6, 2 * CTA_SIZE *sizeof(unsigned int), NULL);
    clStatus |= clEnqueueNDRangeKernel(_context->clQueue, _kernel_FindRadixOffsets, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

    checkCLStatus(clStatus);
}

//#define NUM_BANKS 16
//void clppSort_nvRadixSort::scanNaiveOCL(unsigned int numElements)
//{
//    unsigned int nHist = numElements/2/CTA_SIZE*16;
//    size_t globalWorkSize[1] = {nHist};
//    size_t localWorkSize[1] = {nHist};
//    unsigned int extra_space = nHist / NUM_BANKS;
//    unsigned int shared_mem_size = sizeof(unsigned int) * (nHist + extra_space);
//
//    cl_int clStatus;
//    clStatus  = clSetKernelArg(kernel_ScanNaive, 0, sizeof(cl_mem), (void*)&_clBuffer_CountersSum);
//    clStatus |= clSetKernelArg(kernel_ScanNaive, 1, sizeof(cl_mem), (void*)&_clBuffer_Counters);
//    clStatus |= clSetKernelArg(kernel_ScanNaive, 2, sizeof(unsigned int), (void*)&nHist);
//    clStatus |= clSetKernelArg(kernel_ScanNaive, 3, 2 * shared_mem_size, NULL);
//    clStatus |= clEnqueueNDRangeKernel(_context->clQueue, kernel_ScanNaive, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
//
//    checkCLStatus(clStatus);
//}

void clppSort_nvRadixSort::reorderDataKeysOnlyOCL(unsigned int startbit)
{
    unsigned int totalBlocks = _datasetSize/2/CTA_SIZE;
    size_t globalWorkSize[1] = {CTA_SIZE*totalBlocks};
    size_t localWorkSize[1] = {CTA_SIZE};

    cl_int clStatus;
    clStatus  = clSetKernelArg(_kernel_ReorderDataKeysOnly, 0, sizeof(cl_mem), (void*)&_clBuffer_keys);
    clStatus |= clSetKernelArg(_kernel_ReorderDataKeysOnly, 1, sizeof(cl_mem), (void*)&_clBuffer_TempKeys);
    clStatus |= clSetKernelArg(_kernel_ReorderDataKeysOnly, 2, sizeof(cl_mem), (void*)&_clBuffer_BlockOffsets);
    clStatus |= clSetKernelArg(_kernel_ReorderDataKeysOnly, 3, sizeof(cl_mem), (void*)&_clBuffer_CountersSum);
    clStatus |= clSetKernelArg(_kernel_ReorderDataKeysOnly, 4, sizeof(cl_mem), (void*)&_clBuffer_Counters);
    clStatus |= clSetKernelArg(_kernel_ReorderDataKeysOnly, 5, sizeof(unsigned int), (void*)&startbit);
	clStatus |= clSetKernelArg(_kernel_ReorderDataKeysOnly, 6, sizeof(unsigned int), (void*)&_datasetSize);
    clStatus |= clSetKernelArg(_kernel_ReorderDataKeysOnly, 7, sizeof(unsigned int), (void*)&totalBlocks);
    clStatus |= clSetKernelArg(_kernel_ReorderDataKeysOnly, 8, 2 * CTA_SIZE * sizeof(unsigned int), NULL);
    clStatus |= clEnqueueNDRangeKernel(_context->clQueue, _kernel_ReorderDataKeysOnly, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

    checkCLStatus(clStatus);
}