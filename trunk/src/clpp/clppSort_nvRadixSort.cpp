#include "clpp/clppSort_CPU.h"

#include <algorithm>

#pragma region Construsctor

clppSort_CPU::clppSort_CPU(clppContext* context, string basePath)
{
}

#pragma endregion

#pragma region sort

void clppSort_CPU::sort()
{
    //std::sort((char*)_keys, (char*)_keys + _datasetSize * (_keyBits/8));
    std::sort((int*)_keys, ((int*)_keys) + _datasetSize);
}

#pragma endregion

#pragma region pushDatas

void clppSort_CPU::pushDatas(void* keys, void* values, size_t keySize, size_t valueSize, size_t datasetSize, unsigned int keyBits)
{
    _keys = keys;
	_keySize = keySize;
    _values = values;
    _valueSize = valueSize;
    _datasetSize = datasetSize;
    _keyBits = keyBits;
}

#pragma endregion

#pragma region popDatas

void clppSort_CPU::popDatas()
{
}

#pragma endregion


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

#include "clppSort_nvRadixSort.h"

extern double time1, time2, time3, time4;

clppSort_nvRadixSort::clppSort_nvRadixSort(cl_context GPUContext,
        cl_command_queue CommandQue,
        unsigned int maxElements,
        const char* path,
        const int ctaSize,
        bool keysOnly = true) :
        mNumElements(0),
        mTempValues(0),
        mCounters(0),
        mCountersSum(0),
        mBlockOffsets(0),
        cxGPUContext(GPUContext),
        _context->clQueue(CommandQue),
        CTA_SIZE(ctaSize),
        scan(GPUContext, CommandQue, maxElements/2/CTA_SIZE*16, path)
{

    unsigned int numBlocks = ((maxElements % (CTA_SIZE * 4)) == 0) ?
                             (maxElements / (CTA_SIZE * 4)) : (maxElements / (CTA_SIZE * 4) + 1);
    unsigned int numBlocks2 = ((maxElements % (CTA_SIZE * 2)) == 0) ?
                              (maxElements / (CTA_SIZE * 2)) : (maxElements / (CTA_SIZE * 2) + 1);

    cl_int clStatus;
    d_tempKeys = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(unsigned int) * maxElements, NULL, &clStatus);
    mCounters = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, WARP_SIZE * numBlocks * sizeof(unsigned int), NULL, &clStatus);
    mCountersSum = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, WARP_SIZE * numBlocks * sizeof(unsigned int), NULL, &clStatus);
    mBlockOffsets = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, WARP_SIZE * numBlocks * sizeof(unsigned int), NULL, &clStatus);

    size_t szKernelLength; // Byte size of kernel code
    char *cSourcePath = shrFindFilePath("clppSort_nvRadixSort.cl", path);
    shrCheckError(cSourcePath != NULL, shrTRUE);
    char *cRadixSort = oclLoadProgSource(cSourcePath, "// My comment\n", &szKernelLength);
    oclCheckError(cRadixSort != NULL, shrTRUE);
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cRadixSort, &szKernelLength, &clStatus);
    checkCLStatus(clStatus);
#ifdef MAC
    char *flags = "-DMAC -cl-fast-relaxed-math";
#else
    char *flags = "-cl-fast-relaxed-math";
#endif
    clStatus = clBuildProgram(cpProgram, 0, NULL, flags, NULL, NULL);
    if (clStatus != CL_SUCCESS)
    {
        // write out standard ciErrNumor, Build Log and PTX, then cleanup and exit
        shrLogEx(LOGBOTH | ERRORMSG, clStatus, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "clppSort_nvRadixSort.ptx");
        checkCLStatus(clStatus);
    }

    kernel_RadixSortBlocksKeysOnly = clCreateKernel(cpProgram, "radixSortBlocksKeysOnly", &clStatus);
    checkCLStatus(clStatus);

    kernel_FindRadixOffsets        = clCreateKernel(cpProgram, "findRadixOffsets",        &clStatus);
    checkCLStatus(clStatus);

    kernel_ScanNaive               = clCreateKernel(cpProgram, "scanNaive",               &clStatus);
    checkCLStatus(clStatus);

    kernel_ReorderDataKeysOnly     = clCreateKernel(cpProgram, "reorderDataKeysOnly",     &clStatus);
    checkCLStatus(clStatus);

    free(cRadixSort);
    free(cSourcePath);
}

clppSort_nvRadixSort::~clppSort_nvRadixSort()
{
    clReleaseKernel(kernel_RadixSortBlocksKeysOnly);
    clReleaseKernel(kernel_FindRadixOffsets);
    clReleaseKernel(kernel_ScanNaive);
    clReleaseKernel(kernel_ReorderDataKeysOnly);
    clReleaseProgram(cpProgram);
    clReleaseMemObject(d_tempKeys);
    clReleaseMemObject(mCounters);
    clReleaseMemObject(mCountersSum);
    clReleaseMemObject(mBlockOffsets);
}

// Sorts input arrays of unsigned integer keys and (optional) values
//
// clBuffer_keys    Array of keys for data to be sorted
// elementsCount	Number of elements to be sorted.  Must be <= maxElements passed to the constructor
// keyBits			The number of bits in each key to use for ordering
void clppSort_nvRadixSort::sort(cl_mem clBuffer_keys, unsigned int elementsCount, unsigned int keyBits)
{
	radixSortKeysOnly(clBuffer_keys, elementsCount, keyBits);
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