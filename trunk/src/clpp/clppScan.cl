//------------------------------------------------------------
// Purpose :
// ---------
// Prefix sum or prefix scan is an operation where each output element contains the sum of all input elements preceding it.
//
// Algorithm :
// -----------
// The parallel prefix sum has two principal parts, the reduce phase (also known as the up-sweep phase) and the down-sweep phase.
//
// In the up-sweep reduction phase we traverse the computation tree from bottom to top, computing partial sums.
// After this phase, the last element of the array contains the total sum.
//
// During the down-sweep phase, we traverse the tree from the root and use the partial sums to build the scan in place.
//
// Because the scan pictured is an exclusive sum, a zero is inserted into the last element before the start of the down-sweep phase.
// This zero is then propagated back to the first element.
//
// In our implementation, each compute unit loads and sums up two elements (for the deepest depth). Each subsequent depth during the up-sweep
// phase is processed by half of the compute units from the deeper level and the other way around for the down-sweep phase.
//
// In order to be able to scan large arrays, i.e. arrays that have many more elements than the maximum size of a work-group, the prefix sum has to be decomposed.
// Each work-group computes the prefix scan of its sub-range and outputs a single number representing the sum of all elements in its sub-range.
// The workgroup sums are scanned using exactly the same algorithm.
// When the number of work-group results reaches the size of a work-group, the process is reversed and the work-group sums are
// propagated to the sub-ranges, where each work-group adds the incoming sum to all its elements, thus producing the final scanned array.
//
// References :
// ------------
// NVIDIA Mark Harris. Parallel prefix sum (scan) with CUDA. April 2007
// http://developer.download.nvidia.com/compute/cuda/1_1/Website/projects/scan/doc/scan.pdf
//
// Other references :
// ------------------
// http://developer.download.nvidia.com/compute/cuda/1_1/Website/projects/scan/doc/scan.pdf
// http://developer.nvidia.com/node/57
//------------------------------------------------------------

//#pragma OPENCL EXTENSION cl_amd_printf : enable
#define T int

//------------------------------------------------------------
// kernel__Scan
//
// Purpose : do a scan 
//------------------------------------------------------------

__kernel
void kernel__ExclusivePrefixScan(
	__global const T* values,
	__global T* valuesOut,
	__local T* localBuffer,
	__global T* blockSums,
	const uint N
	)
{
    const uint tid = get_local_id(0);
    const uint groupSize = get_local_size(0);
    const uint blockSize = groupSize << 1;
	const uint groupId = get_group_id(0);
    const uint globalOffset = groupId * blockSize;	
    int offset = 1;
    const int tid2_0 = tid << 1; // 2 * tid
    const int tid2_1 = tid2_0 + 1;

	localBuffer[tid2_0] = (tid2_0 + globalOffset < N) ? values[tid2_0 + globalOffset] : 0;
	localBuffer[tid2_1] = (tid2_1 + globalOffset < N) ? values[tid2_1 + globalOffset] : 0;
	
    // bottom-up
    for(uint d = groupSize; d > 0; d >>= 1)
	{
        barrier(CLK_LOCAL_MEM_FENCE);
        if(tid < d)
		{
            const uint ai = mad24(offset, (tid2_1+0), -1);	// offset*(tid2_0+1)-1 = offset*(tid2_1+0)-1
            const uint bi = mad24(offset, (tid2_1+1), -1);	// offset*(tid2_1+1)-1;
            localBuffer[bi] += localBuffer[ai];
        }
        offset <<= 1;
    }

    //barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 1)
	{
        blockSums[groupId] = localBuffer[blockSize-1];
        localBuffer[blockSize-1] = 0;
    }

    // top-down
    for(uint d = 1; d < blockSize; d <<= 1)
	{
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
        if(tid < d)
		{
            const uint ai = mad24(offset, (tid2_1+0), -1); // offset*(tid2_0+1)-1 = offset*(tid2_1+0)-1
            const uint bi = mad24(offset, (tid2_1+1), -1); // offset*(tid2_1+1)-1;
            T tmp = localBuffer[ai];
            localBuffer[ai] = localBuffer[bi];
            localBuffer[bi] += tmp;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Write out
    if (tid2_0 + globalOffset < N)
        valuesOut[tid2_0 + globalOffset] = localBuffer[tid2_0];
		
    if (tid2_1 + globalOffset < N)
        valuesOut[tid2_1 + globalOffset] = localBuffer[tid2_1];
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

    __local T localBuffer[1];

    if (tid == 0)
        localBuffer[0] = blockSums[blockId];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < N)
        memOut[gid] += localBuffer[0];
		
    if (gid + 1 < N)
        memOut[gid + 1] += localBuffer[0];
}
