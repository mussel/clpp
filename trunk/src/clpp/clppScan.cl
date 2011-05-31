// References :
//
// http://developer.download.nvidia.com/compute/cuda/1_1/Website/projects/scan/doc/scan.pdf
// http://developer.nvidia.com/node/57

//#pragma OPENCL EXTENSION cl_amd_printf : enable
#define T int

//------------------------------------------------------------
// kernel__Scan
//
// Purpose :
// Prefix sum or prefix scan is an operation where each output element contains the sum of all input elements preceding it.
//------------------------------------------------------------

__kernel
void kernel__ExclusivePrefixScan(
	__global const T* values,
	__global T* valuesOut,
	//__global T* shared,
	__local T* shared,
	__global T* blockSums,
	const uint N
	)
{
	//__local T shared[1024];
	
    const uint tid = get_local_id(0);
    const uint groupSize = get_local_size(0);
    const uint blockSize = groupSize << 1;
	const uint groupId = get_group_id(0);
    const uint globalOffset = groupId * blockSize;	
    int offset = 1;
    const int tid2_0 = tid << 1; // 2 * tid
    const int tid2_1 = tid2_0 + 1;

	shared[tid2_0] = (tid2_0 + globalOffset < N) ? values[tid2_0 + globalOffset] : 0;
	shared[tid2_1] = (tid2_1 + globalOffset < N) ? values[tid2_1 + globalOffset] : 0;

    // bottom-up
    for(uint d = groupSize; d > 0; d >>= 1)
	{
        barrier(CLK_LOCAL_MEM_FENCE);
        if(tid < d)
		{
            const uint ai = mad24(offset, (tid2_1+0), -1);	// offset*(tid2_0+1)-1 = offset*(tid2_1+0)-1
            const uint bi = mad24(offset, (tid2_1+1), -1);	// offset*(tid2_1+1)-1;
            shared[bi] += shared[ai];
        }
        offset <<= 1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == 0)
	{
        blockSums[groupId] = shared[blockSize-1];
        shared[blockSize-1] = 0;
    }

    // top-down
    for(uint d = 1; d < blockSize; d <<= 1)
	{
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if(tid < d)
		{
            const uint ai = mad24(offset, (tid2_1+0), -1); // offset*(tid2_0+1)-1 = offset*(tid2_1+0)-1
            const uint bi = mad24(offset, (tid2_1+1), -1); //offset*(tid2_1+1)-1;
            float tmp = shared[ai];
            shared[ai] = shared[bi];
            shared[bi] += tmp;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Write out
    if (tid2_0 + globalOffset < N)
        valuesOut[tid2_0 + globalOffset] = shared[tid2_0];
		
    if (tid2_1 + globalOffset < N)
        valuesOut[tid2_1 + globalOffset] = shared[tid2_1];
}

__kernel
void kernel__UniformAdd(
	__global T* memOut,
	__global const T* blockSums,
	const uint N
	)
{
    const uint gid = get_global_id(0) * 2;
    const uint tid = get_local_id(0);
    const uint blockId = get_group_id(0);

    __local T shared[1];

    if (tid == 0)
        shared[0] = blockSums[blockId];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < N)
        memOut[gid] += shared[0];
		
    if (gid + 1 < N)
        memOut[gid + 1] += shared[0];
}
