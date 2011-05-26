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

#include "clpp/clppSort_nvScan.h"
#include "clpp/clppSort.h"

clppSort_nvScan::clppSort_nvScan(clppContext* context, string basePath, unsigned int numElements) : mNumElements(numElements)
{
	_context = context;

    cl_int clStatus;
    if (numElements > MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE)
    {
		d_Buffer = clCreateBuffer(context->clContext, CL_MEM_READ_WRITE, numElements / MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE * sizeof(cl_uint), NULL, &clStatus);
        clppSort::checkCLStatus(clStatus);
    }

	if (!compile(context, basePath, "clppSort_nvScan_b.cl"))
		return;

	// Prepare the kernels
    ckScanExclusiveLocal1 = clCreateKernel(_clProgram, "scanExclusiveLocal1", &clStatus);
    checkCLStatus(clStatus);

    ckScanExclusiveLocal2 = clCreateKernel(_clProgram, "scanExclusiveLocal2", &clStatus);
    checkCLStatus(clStatus);

    ckUniformUpdate = clCreateKernel(_clProgram, "uniformUpdate", &clStatus);
    checkCLStatus(clStatus);
}

clppSort_nvScan::~clppSort_nvScan()
{
    cl_int clStatus;

    clStatus  = clReleaseKernel(ckScanExclusiveLocal1);
    clStatus |= clReleaseKernel(ckScanExclusiveLocal2);
    clStatus |= clReleaseKernel(ckUniformUpdate);
    if (mNumElements > MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE)
        clStatus |= clReleaseMemObject(d_Buffer);

    clStatus |= clReleaseProgram(_clProgram);
    clppSort::checkCLStatus(clStatus);
}

// main exclusive scan routine
void clppSort_nvScan::scanExclusiveLarge(cl_mem d_Dst, cl_mem d_Src, unsigned int batchSize, unsigned int arrayLength)
{
    //Check power-of-two factorization
    unsigned int log2L;
    unsigned int factorizationRemainder = factorRadix2(log2L, arrayLength);

    scanExclusiveLocal1(d_Dst, d_Src, (batchSize * arrayLength) / (4 * WORKGROUP_SIZE), 4 * WORKGROUP_SIZE);

    scanExclusiveLocal2( d_Buffer, d_Dst, d_Src, batchSize, arrayLength / (4 * WORKGROUP_SIZE));

    uniformUpdate(d_Dst, d_Buffer, (batchSize * arrayLength) / (4 * WORKGROUP_SIZE));
}

void clppSort_nvScan::scanExclusiveLocal1(cl_mem d_Dst, cl_mem d_Src, unsigned int n, unsigned int size)
{
    cl_int clStatus;
    size_t localWorkSize, globalWorkSize;

    clStatus  = clSetKernelArg(ckScanExclusiveLocal1, 0, sizeof(cl_mem), (void *)&d_Dst);
    clStatus |= clSetKernelArg(ckScanExclusiveLocal1, 1, sizeof(cl_mem), (void *)&d_Src);
    clStatus |= clSetKernelArg(ckScanExclusiveLocal1, 2, 2 * WORKGROUP_SIZE * sizeof(unsigned int), NULL);
    clStatus |= clSetKernelArg(ckScanExclusiveLocal1, 3, sizeof(unsigned int), (void *)&size);
    clppSort::checkCLStatus(clStatus);

    localWorkSize = WORKGROUP_SIZE;
    globalWorkSize = (n * size) / 4;

	clStatus = clEnqueueNDRangeKernel(_context->clQueue, ckScanExclusiveLocal1, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    clppSort::checkCLStatus(clStatus);
}

void clppSort_nvScan::scanExclusiveLocal2(cl_mem d_Buffer, cl_mem d_Dst, cl_mem d_Src, unsigned int n, unsigned int size)
{
    cl_int clStatus;
    size_t localWorkSize, globalWorkSize;

    unsigned int elements = n * size;
    clStatus  = clSetKernelArg(ckScanExclusiveLocal2, 0, sizeof(cl_mem), (void *)&d_Buffer);
    clStatus |= clSetKernelArg(ckScanExclusiveLocal2, 1, sizeof(cl_mem), (void *)&d_Dst);
    clStatus |= clSetKernelArg(ckScanExclusiveLocal2, 2, sizeof(cl_mem), (void *)&d_Src);
    clStatus |= clSetKernelArg(ckScanExclusiveLocal2, 3, 2 * WORKGROUP_SIZE * sizeof(unsigned int), NULL);
    clStatus |= clSetKernelArg(ckScanExclusiveLocal2, 4, sizeof(unsigned int), (void *)&elements);
    clStatus |= clSetKernelArg(ckScanExclusiveLocal2, 5, sizeof(unsigned int), (void *)&size);
    clppSort::checkCLStatus(clStatus);

    localWorkSize = WORKGROUP_SIZE;
    globalWorkSize = iSnapUp(elements, WORKGROUP_SIZE);

    clStatus = clEnqueueNDRangeKernel(_context->clQueue, ckScanExclusiveLocal2, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    clppSort::checkCLStatus(clStatus);
}

void clppSort_nvScan::uniformUpdate(cl_mem d_Dst, cl_mem d_Buffer, unsigned int n)
{
    cl_int clStatus;
    size_t localWorkSize, globalWorkSize;

    clStatus  = clSetKernelArg(ckUniformUpdate, 0, sizeof(cl_mem), (void *)&d_Dst);
    clStatus |= clSetKernelArg(ckUniformUpdate, 1, sizeof(cl_mem), (void *)&d_Buffer);
    clppSort::checkCLStatus(clStatus);

    localWorkSize = WORKGROUP_SIZE;
    globalWorkSize = n * WORKGROUP_SIZE;

    clStatus = clEnqueueNDRangeKernel(_context->clQueue, ckUniformUpdate, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
	checkCLStatus(clStatus);
}
