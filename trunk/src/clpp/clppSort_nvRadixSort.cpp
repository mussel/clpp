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
        mCounters(0),
        mCountersSum(0),
        mBlockOffsets(0),
        CTA_SIZE(ctaSize),
        scan(context, basePath, maxElements/2/CTA_SIZE*16)
{
	cl_int clStatus;

    unsigned int numBlocks = ((maxElements % (CTA_SIZE * 4)) == 0) ?
                             (maxElements / (CTA_SIZE * 4)) : (maxElements / (CTA_SIZE * 4) + 1);
    unsigned int numBlocks2 = ((maxElements % (CTA_SIZE * 2)) == 0) ?
                              (maxElements / (CTA_SIZE * 2)) : (maxElements / (CTA_SIZE * 2) + 1);

	d_tempKeys = clCreateBuffer(context->clContext, CL_MEM_READ_WRITE, sizeof(unsigned int) * maxElements, NULL, &clStatus);
	mCounters = clCreateBuffer(context->clContext, CL_MEM_READ_WRITE, WARP_SIZE * numBlocks * sizeof(unsigned int), NULL, &clStatus);
	mCountersSum = clCreateBuffer(context->clContext, CL_MEM_READ_WRITE, WARP_SIZE * numBlocks * sizeof(unsigned int), NULL, &clStatus);
	mBlockOffsets = clCreateBuffer(context->clContext, CL_MEM_READ_WRITE, WARP_SIZE * numBlocks * sizeof(unsigned int), NULL, &clStatus); 

	if (!compile(context, basePath, "clppSort_nvRadixSort.cl"))
		return;

	//---- Prepare all the kernels
	kernel_RadixSortBlocksKeysOnly = clCreateKernel(_clProgram, "radixSortBlocksKeysOnly", &clStatus);
    checkCLStatus(clStatus);

    kernel_FindRadixOffsets = clCreateKernel(_clProgram, "findRadixOffsets", &clStatus);
    checkCLStatus(clStatus);

    kernel_ScanNaive = clCreateKernel(_clProgram, "scanNaive", &clStatus);
    checkCLStatus(clStatus);

    kernel_ReorderDataKeysOnly = clCreateKernel(_clProgram, "reorderDataKeysOnly", &clStatus);
    checkCLStatus(clStatus);
}

clppSort_nvRadixSort::~clppSort_nvRadixSort()
{
    clReleaseKernel(kernel_RadixSortBlocksKeysOnly);
    clReleaseKernel(kernel_FindRadixOffsets);
    clReleaseKernel(kernel_ScanNaive);
    clReleaseKernel(kernel_ReorderDataKeysOnly);
    
    clReleaseMemObject(d_tempKeys);
    clReleaseMemObject(mCounters);
    clReleaseMemObject(mCountersSum);
    clReleaseMemObject(mBlockOffsets);
}

void clppSort_nvRadixSort::pushDatas(cl_mem clBuffer_keys, cl_mem clBuffer_values, size_t datasetSize, unsigned int keyBits)
{
	cl_int clStatus;

	if (_keys != 0)
	{
		clStatus = clEnqueueWriteBuffer(_context->clQueue, clBuffer_keys, CL_FALSE, 0, _keySize * _datasetSize, _keys, 0, NULL, NULL);
		checkCLStatus(clStatus);
	}
}

void clppSort_nvRadixSort::popDatas()
{
	cl_int clStatus;

    clFinish(_context->clQueue);     // wait end of read

	clStatus = clEnqueueReadBuffer(_context->clQueue, _clBuffer_keys, CL_TRUE, 0, _keySize * _datasetSize, _keys, 0, NULL, NULL);
	checkCLStatus(clStatus);
}

void clppSort_nvRadixSort::sort()
{
	radixSortKeysOnly(_clBuffer_keys, _datasetSize, _keyBits);
}

void clppSort_nvRadixSort::radixSortKeysOnly(cl_mem d_keys, unsigned int numElements, unsigned int keyBits)
{
    int i = 0;
    while (keyBits > i*bitStep)
    {
        radixSortStepKeysOnly(d_keys, bitStep, i*bitStep, numElements);
        i++;
    }
}

void clppSort_nvRadixSort::radixSortStepKeysOnly(cl_mem d_keys, unsigned int nbits, unsigned int startbit, unsigned int numElements)
{
    // Four step algorithms from Satish, Harris & Garland
    radixSortBlocksKeysOnlyOCL(d_keys, nbits, startbit, numElements);

    findRadixOffsetsOCL(startbit, numElements);

    scan.scanExclusiveLarge(mCountersSum, mCounters, 1, numElements / 2 / CTA_SIZE*16);

    reorderDataKeysOnlyOCL(d_keys, startbit, numElements);
}

void clppSort_nvRadixSort::radixSortBlocksKeysOnlyOCL(cl_mem d_keys, unsigned int nbits, unsigned int startbit, unsigned int numElements)
{
    unsigned int totalBlocks = numElements/4/CTA_SIZE;
    size_t globalWorkSize[1] = {CTA_SIZE*totalBlocks};
    size_t localWorkSize[1] = {CTA_SIZE};

    cl_int clStatus;
    clStatus  = clSetKernelArg(kernel_RadixSortBlocksKeysOnly, 0, sizeof(cl_mem), (void*)&d_keys);
    clStatus |= clSetKernelArg(kernel_RadixSortBlocksKeysOnly, 1, sizeof(cl_mem), (void*)&d_tempKeys);
    clStatus |= clSetKernelArg(kernel_RadixSortBlocksKeysOnly, 2, sizeof(unsigned int), (void*)&nbits);
    clStatus |= clSetKernelArg(kernel_RadixSortBlocksKeysOnly, 3, sizeof(unsigned int), (void*)&startbit);
    clStatus |= clSetKernelArg(kernel_RadixSortBlocksKeysOnly, 4, sizeof(unsigned int), (void*)&numElements);
    clStatus |= clSetKernelArg(kernel_RadixSortBlocksKeysOnly, 5, sizeof(unsigned int), (void*)&totalBlocks);
    clStatus |= clSetKernelArg(kernel_RadixSortBlocksKeysOnly, 6, 4*CTA_SIZE*sizeof(unsigned int), NULL);
    clStatus |= clEnqueueNDRangeKernel(_context->clQueue, kernel_RadixSortBlocksKeysOnly, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

    checkCLStatus(clStatus);
}

void clppSort_nvRadixSort::findRadixOffsetsOCL(unsigned int startbit, unsigned int numElements)
{
    unsigned int totalBlocks = numElements/2/CTA_SIZE;
    size_t globalWorkSize[1] = {CTA_SIZE*totalBlocks};
    size_t localWorkSize[1] = {CTA_SIZE};
    
	cl_int clStatus;
    clStatus  = clSetKernelArg(kernel_FindRadixOffsets, 0, sizeof(cl_mem), (void*)&d_tempKeys);
    clStatus |= clSetKernelArg(kernel_FindRadixOffsets, 1, sizeof(cl_mem), (void*)&mCounters);
    clStatus |= clSetKernelArg(kernel_FindRadixOffsets, 2, sizeof(cl_mem), (void*)&mBlockOffsets);
    clStatus |= clSetKernelArg(kernel_FindRadixOffsets, 3, sizeof(unsigned int), (void*)&startbit);
    clStatus |= clSetKernelArg(kernel_FindRadixOffsets, 4, sizeof(unsigned int), (void*)&numElements);
    clStatus |= clSetKernelArg(kernel_FindRadixOffsets, 5, sizeof(unsigned int), (void*)&totalBlocks);
    clStatus |= clSetKernelArg(kernel_FindRadixOffsets, 6, 2 * CTA_SIZE *sizeof(unsigned int), NULL);
    clStatus |= clEnqueueNDRangeKernel(_context->clQueue, kernel_FindRadixOffsets, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

    checkCLStatus(clStatus);
}

#define NUM_BANKS 16
void clppSort_nvRadixSort::scanNaiveOCL(unsigned int numElements)
{
    unsigned int nHist = numElements/2/CTA_SIZE*16;
    size_t globalWorkSize[1] = {nHist};
    size_t localWorkSize[1] = {nHist};
    unsigned int extra_space = nHist / NUM_BANKS;
    unsigned int shared_mem_size = sizeof(unsigned int) * (nHist + extra_space);

    cl_int clStatus;
    clStatus  = clSetKernelArg(kernel_ScanNaive, 0, sizeof(cl_mem), (void*)&mCountersSum);
    clStatus |= clSetKernelArg(kernel_ScanNaive, 1, sizeof(cl_mem), (void*)&mCounters);
    clStatus |= clSetKernelArg(kernel_ScanNaive, 2, sizeof(unsigned int), (void*)&nHist);
    clStatus |= clSetKernelArg(kernel_ScanNaive, 3, 2 * shared_mem_size, NULL);
    clStatus |= clEnqueueNDRangeKernel(_context->clQueue, kernel_ScanNaive, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

    checkCLStatus(clStatus);
}

void clppSort_nvRadixSort::reorderDataKeysOnlyOCL(cl_mem d_keys, unsigned int startbit, unsigned int numElements)
{
    unsigned int totalBlocks = numElements/2/CTA_SIZE;
    size_t globalWorkSize[1] = {CTA_SIZE*totalBlocks};
    size_t localWorkSize[1] = {CTA_SIZE};

    cl_int clStatus;
    clStatus  = clSetKernelArg(kernel_ReorderDataKeysOnly, 0, sizeof(cl_mem), (void*)&d_keys);
    clStatus |= clSetKernelArg(kernel_ReorderDataKeysOnly, 1, sizeof(cl_mem), (void*)&d_tempKeys);
    clStatus |= clSetKernelArg(kernel_ReorderDataKeysOnly, 2, sizeof(cl_mem), (void*)&mBlockOffsets);
    clStatus |= clSetKernelArg(kernel_ReorderDataKeysOnly, 3, sizeof(cl_mem), (void*)&mCountersSum);
    clStatus |= clSetKernelArg(kernel_ReorderDataKeysOnly, 4, sizeof(cl_mem), (void*)&mCounters);
    clStatus |= clSetKernelArg(kernel_ReorderDataKeysOnly, 5, sizeof(unsigned int), (void*)&startbit);
    clStatus |= clSetKernelArg(kernel_ReorderDataKeysOnly, 6, sizeof(unsigned int), (void*)&numElements);
    clStatus |= clSetKernelArg(kernel_ReorderDataKeysOnly, 7, sizeof(unsigned int), (void*)&totalBlocks);
    clStatus |= clSetKernelArg(kernel_ReorderDataKeysOnly, 8, 2 * CTA_SIZE * sizeof(unsigned int), NULL);
    clStatus |= clEnqueueNDRangeKernel(_context->clQueue, kernel_ReorderDataKeysOnly, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

    checkCLStatus(clStatus);
}