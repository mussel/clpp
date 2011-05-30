#include "clpp/clppSort_Merill.h"

#pragma region OpenCL simulation

#define __kernel
#define __global
#define __local
#define __read_only
#define barrier(A)
#define uint unsigned int
#define INFINITY 0xFFFFFFFF

static unsigned int _gid;
static unsigned int _lid;
static unsigned int _ls;
unsigned int get_global_id(int l) { return _gid; }
unsigned int get_local_id(int l) { return _lid; }
unsigned int get_local_size(int l) { return _ls; }

#pragma endregion

__kernel
void kernel_float_scan(
	__global __read_only float* inputs,
	__global __read_only float* outputs,
	__local __read_only float* temp,
	uint n
	);

#pragma region Constructor

clppSort_Merill::clppSort_Merill(clppContext* context, unsigned int maxElements)
{
}

#pragma endregion

#pragma region sort

void clppSort_Merill::sort()
{
	float inputs[] = {1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5};
	float outputs[20];
	float temp[20];

	//---- Simulate 'clEnqueueNDRangeKernel' 
	int global_work_size = 20;
	int local_work_size = 4;

	_gid = 0;
	_ls = local_work_size;
	int groupsCount = global_work_size / local_work_size;
	for(int g = 0; g < groupsCount; g++)
	{
		_lid = 0;
		for(int lw = 0; lw < local_work_size; lw++)
		{
			kernel_float_scan(inputs, outputs, temp, 20);
			_gid++;
			_lid++;
		}
	}
}

#pragma endregion

#pragma region pushDatas

void clppSort_Merill::pushDatas(void* keys, void* values, size_t keySize, size_t valueSize, size_t datasetSize, unsigned int keyBits)
{
	_keys = keys;
	_keySize = keySize;
	_values = values;
	_valueSize = valueSize;
	_datasetSize = datasetSize;
	_keyBits = keyBits;
}

void clppSort_Merill::pushDatas(cl_mem clBuffer_keys, cl_mem clBuffer_values, size_t datasetSize, unsigned int keyBits) 
{
}

#pragma endregion

#pragma region popDatas

void clppSort_Merill::popDatas()
{
}

#pragma endregion

__kernel
void kernel_float_scan(
	__global __read_only float* inputs,
	__global __read_only float* outputs,
	__local __read_only float* temp,
	uint n
	)
{
	size_t gid = get_global_id(0);
	size_t lid = get_local_id(0);
	
	if (gid < n)
		temp[gid] = inputs[gid];
	else
		// Infinity is the identity element for the min operation
		temp[gid] = INFINITY;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	#pragma unroll
	for(uint offset = 1; offset < get_local_size(0); offset <<= 1)
	{
	
		// Iteration 1 : will be true for every other work-item in the work-group.
		// Iteration 2 : will be true for every 4th work-item in the work group.
		// Iteration 3 : will be true for every 8th work-item in the work group.
		int mask = (offset << 1) - 1; // 1 3 7 15 31 => 1 11 111 1111 11111
		if ((lid & mask) == 0)
	
		temp[gid] += temp[gid+offset];
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
	}
	
	outputs[gid] = temp[gid];
}