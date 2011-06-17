//------------------------------------------------------------
// Purpose :
// ---------
//
// Algorithm :
// -----------
// Radix sort algorithm for key-value pairs. This work is based on the Blelloch
// paper and optimized with the technique described in the Satish/Harris/Garland paper.
//
// References :
// ------------
// Designing Efficient Sorting Algorithms for Manycore GPUs. Nadathur Satish, Mark Harris, Michael Garland. http://mgarland.org/files/papers/gpusort-ipdps09.pdf
// http://www.sci.utah.edu/~csilva/papers/cgf.pdf
// Radix Sort For Vector Multiprocessors, Marco Zagha and Guy E. Blelloch
//------------------------------------------------------------

#define LTYPE int	// TODO

#define MAX_INT2 (int2)0x7FFFFFFF

// Too slow because we use a workgroup=32. So, it seems that we are unable to fill the GPU.

#ifdef OCL_DEVICE_GPU

// Exclusive scan of 32 elements by using the SIMT capability (to avoid synchronization of work items).
// Directly do it for 4x32 elements, simply use an offset
inline void scan_simt_exclusive_4(__local LTYPE* input, size_t tid1, const uint lane)
{
	const uint tid2 = tid1 + 32;
	const uint tid3 = tid2 + 32;
	const uint tid4 = tid3 + 32;
	
	if (lane > 0 )
	{
		input[tid1] += input[tid1 - 1];
		input[tid2] += input[tid2 - 1];
		input[tid3] += input[tid3 - 1];
		input[tid4] += input[tid4 - 1];
	}
	
	if (lane > 1 )
	{
		input[tid1] += input[tid1 - 2];
		input[tid2] += input[tid2 - 2];
		input[tid3] += input[tid3 - 2];
		input[tid4] += input[tid4 - 2];
	}
	
	if (lane > 3 )
	{
		input[tid1] += input[tid1 - 4];
		input[tid2] += input[tid2 - 4];
		input[tid3] += input[tid3 - 4];
		input[tid4] += input[tid4 - 4];
	}
	
	if (lane > 7 )
	{
		input[tid1] += input[tid1 - 8];
		input[tid2] += input[tid2 - 8];
		input[tid3] += input[tid3 - 8];
		input[tid4] += input[tid4 - 8];
	}
	
	if (lane > 15)
	{
		input[tid1] += input[tid1 - 16];
		input[tid2] += input[tid2 - 16];
		input[tid3] += input[tid3 - 16];
		input[tid4] += input[tid4 - 16];
	}
}

inline 
void exclusive_scan_128(const uint tid, const int4 tid4, uint blockSize, __local LTYPE* localBuffer, __local LTYPE* incSum, uint size)
{
	const uint lane = tid & 31;
	
	scan_simt_exclusive_4(localBuffer, tid, lane);
	
	barrier(CLK_LOCAL_MEM_FENCE);	 // NOT NECESSARY ?
	
	//barrier(CLK_LOCAL_MEM_FENCE);
	
	__local int sum[3];
	if (lane < 1)
	{
		sum[0] = localBuffer[tid + blockSize - 1];
		sum[1] = sum[0] + localBuffer[tid + 2 * blockSize - 1];
		sum[2] = sum[1] + localBuffer[tid + 3 * blockSize - 1];
	}
	
	if (tid == blockSize-1)
        incSum[0] = localBuffer[tid4.w];
		
	barrier(CLK_LOCAL_MEM_FENCE);
		
	localBuffer[tid + blockSize] 		+= sum[0];
	localBuffer[tid + 2 * blockSize] 	+= sum[1];
	localBuffer[tid + 3 * blockSize] 	+= sum[2];
}

#else

inline
void exclusive_scan_4(const uint tid, const int4 tid4, uint blockSize, __local LTYPE* localBuffer, __local LTYPE* incSum)
{
    const int tid2_0 = tid << 1;
    const int tid2_1 = tid2_0 + 1;
	
	int offset = 4;
	//#pragma unroll
	for (uint d = 16; d > 0; d >>= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
		
        if (tid < d)
        {
            const uint ai = mad24(offset, (tid2_1+0), -1);	// offset*(tid2_0+1)-1 = offset*(tid2_1+0)-1
            const uint bi = mad24(offset, (tid2_1+1), -1);	// offset*(tid2_1+1)-1;
			
            localBuffer[bi] += localBuffer[ai];
        }
		
		offset <<= 1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid == blockSize-1)
    {
        incSum[0] = localBuffer[tid4.w];
        localBuffer[tid4.w] = 0;
    }

    // expansion
	//#pragma unroll
    for (uint d = 1; d < 32; d <<= 1)
	//for (int d = 1; d < blockSize; d <<= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
		offset >>= 1;
		
        if (tid < d)
        {
            const uint ai = mad24(offset, (tid2_1+0), -1); // offset*(tid2_0+1)-1 = offset*(tid2_1+0)-1
            const uint bi = mad24(offset, (tid2_1+1), -1); // offset*(tid2_1+1)-1;
			
            LTYPE tmp = localBuffer[ai];
            localBuffer[ai] = localBuffer[bi];
            localBuffer[bi] += tmp;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

inline 
void exclusive_scan_128(const uint tid, const int4 tid4, uint blockSize, __local LTYPE* localBuffer, __local LTYPE* incSum, uint size)
{
	// local serial reduction
	localBuffer[tid4.y] += localBuffer[tid4.x];
	localBuffer[tid4.w] += localBuffer[tid4.z];
	localBuffer[tid4.w] += localBuffer[tid4.y];
		
	// Exclusive scan starting with an offset of 4
	exclusive_scan_4(tid, tid4, blockSize, localBuffer, incSum);
		
	// local 4-element serial expansion
	LTYPE tmp;
	tmp = localBuffer[tid4.y];    localBuffer[tid4.y] = localBuffer[tid4.w];  localBuffer[tid4.w] += tmp;
	tmp = localBuffer[tid4.x];    localBuffer[tid4.x] = localBuffer[tid4.y];  localBuffer[tid4.y] += tmp;
	tmp = localBuffer[tid4.z];    localBuffer[tid4.z] = localBuffer[tid4.w];  localBuffer[tid4.w] += tmp;
}

#endif

//------------------------------------------------------------
// kernel__radixLocalSort
//
// Purpose :
// 1) Each workgroup sorts its tile by using local memory
// 2) Create an histogram of d=2^b digits entries
//------------------------------------------------------------

__kernel
void kernel__radixLocalSort(
	__local int2* shared,      // size 4*4 int2s (8 kB)
	__local LTYPE* indices,    // size 4*4 shorts (4 kB)
	__local LTYPE* sharedSum,  // size 4*4*2 shorts (2 kB)
	__global int2* data,       // size 4*4 int2s per block (8 kB)
	__global int* hist,        // size 16  per block (64 B)
	__global int* blockHists,  // size 16 int2s per block (64 B)
	const int bitOffset,       // k*4, k=0..7
	const int N)               // N = 32 (32x int2 global)
{
    const int gid4 = (int)(get_global_id(0) << 2);
    const int tid = (int)get_local_id(0);
    const int4 tid4 = (int4)(tid << 2) + (const int4)(0,1,2,3);
    const int blockId = (int)get_group_id(0);
    const int blockSize = (int)get_local_size(0);
    const int blockSize4 = (blockSize << 2);

    __local int localHistStart[16];
    __local int localHistEnd[16];
    __local LTYPE incSum[1];

    // Each thread copies 4 (Cell,Tri) pairs into local memory
    shared[tid4.x] = (gid4+0 < N) ? data[gid4+0] : MAX_INT2;
    shared[tid4.y] = (gid4+1 < N) ? data[gid4+1] : MAX_INT2;
    shared[tid4.z] = (gid4+2 < N) ? data[gid4+2] : MAX_INT2;
    shared[tid4.w] = (gid4+3 < N) ? data[gid4+3] : MAX_INT2;

    indices[tid4.x] = tid4.x;
    indices[tid4.y] = tid4.y;
    indices[tid4.z] = tid4.z;
    indices[tid4.w] = tid4.w;

    int srcBase = 0;
    int dstBase = blockSize4;
	
	//-------- 1) 4 x local 1-bit split
	// Should find a way to use createBestScan to improve performance here

    uint shift = bitOffset;
    for(uint i = 0; i < 4; i++, shift++)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        sharedSum[tid4.x] = (LTYPE)1 - ((LTYPE)(((int)shared[indices[srcBase + tid4.x]].x) >> shift) & 0x1);
        sharedSum[tid4.y] = (LTYPE)1 - ((LTYPE)(((int)shared[indices[srcBase + tid4.y]].x) >> shift) & 0x1);
        sharedSum[tid4.z] = (LTYPE)1 - ((LTYPE)(((int)shared[indices[srcBase + tid4.z]].x) >> shift) & 0x1);
        sharedSum[tid4.w] = (LTYPE)1 - ((LTYPE)(((int)shared[indices[srcBase + tid4.w]].x) >> shift) & 0x1);
		
		// Do a scan of the 128 bits
		exclusive_scan_128(tid, tid4, blockSize, sharedSum, incSum, N);

        // Permutations
        for(uint b = 0; b < 4; b++)
        {
            uint idx = tid4.x + b;
            barrier(CLK_LOCAL_MEM_FENCE);
			
            int flag = (((int)shared[indices[srcBase + idx]].x) >> shift) & 0x1;
			
            //if (flag == 1)
            //    indices[dstBase + (int)incSum[0] + idx - (int)sharedSum[idx]] = indices[srcBase + idx];
            //else
            //    indices[dstBase + (int)sharedSum[idx]] = indices[srcBase + idx];
				
			// Faster version for GPU (no divergence)
			uint targetOffset = flag * ( (int)incSum[0] + idx - (int)sharedSum[idx] ) + (1-flag) * ((int)sharedSum[idx]);
			indices[dstBase + targetOffset] = indices[srcBase + idx];
        }

        // Pingpong left and right halves of the indirection buffer
        int tmpBase = srcBase;
		srcBase = dstBase;
		dstBase = tmpBase;
    }
	
    barrier(CLK_LOCAL_MEM_FENCE);
	
    // Write sorted data back to global mem
    if (gid4+0 < N) data[gid4+0] = shared[indices[srcBase + tid4.x]];
    if (gid4+1 < N) data[gid4+1] = shared[indices[srcBase + tid4.y]];
    if (gid4+2 < N) data[gid4+2] = shared[indices[srcBase + tid4.z]];
    if (gid4+3 < N) data[gid4+3] = shared[indices[srcBase + tid4.w]];
	
	//-------- 2) Histogram

    // init histogram values
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16)
    {
        localHistStart[tid] = 0;
        localHistEnd[tid] = -1;
    }

    // Start computation
	
    barrier(CLK_LOCAL_MEM_FENCE);
    int ka, kb;
    if (tid4.x > 0)
    {
        ka = (((int)shared[indices[srcBase + tid4.x-0]].x) >> bitOffset) & 0xF;
        kb = (((int)shared[indices[srcBase + tid4.x-1]].x) >> bitOffset) & 0xF;
        if (ka != kb)
        {
            localHistEnd[kb] = tid4.x - 1;
            localHistStart[ka] = tid4.x;
        }
    }

    ka = (((int)shared[indices[srcBase + tid4.y]].x) >> bitOffset) & 0xF;
    kb = (((int)shared[indices[srcBase + tid4.x]].x) >> bitOffset) & 0xF;
    if (ka != kb)
    {
        localHistEnd[kb] = tid4.x;
        localHistStart[ka] = tid4.y;
    }

    ka = (((int)shared[indices[srcBase + tid4.z]].x) >> bitOffset) & 0xF;
    kb = (((int)shared[indices[srcBase + tid4.y]].x) >> bitOffset) & 0xF;
    if (ka != kb)
    {
        localHistEnd[kb] = tid4.y;
        localHistStart[ka] = tid4.z;
    }

    ka = (((int)shared[indices[srcBase + tid4.w]].x) >> bitOffset) & 0xF;
    kb = (((int)shared[indices[srcBase + tid4.z]].x) >> bitOffset) & 0xF;
    if (ka != kb)
    {
        localHistEnd[kb] = tid4.z;
        localHistStart[ka] = tid4.w;
    }

    if (tid == 0)
    {
        localHistEnd[(((int)shared[indices[srcBase + blockSize4-1]].x) >> bitOffset) & 0xF] = blockSize4 - 1;
        localHistStart[(((int)shared[indices[srcBase]].x) >> bitOffset) & 0xF] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Write histogram to global memomry
    const int numBlocks = get_num_groups(0);
    if (tid < 16)
    {
        hist[tid * numBlocks + blockId] = localHistEnd[tid] - localHistStart[tid] + 1;
        blockHists[(blockId << 5) + tid] = localHistStart[tid];
    }
}

//------------------------------------------------------------
// kernel__radixPermute
//
// Purpose : Prefix sum results are used to scatter each work-group's elements to their correct position.
//------------------------------------------------------------

__kernel
void kernel__radixPermute(
	__global const int2* dataIn,		// size 4*4 int2s per block
	__global int2* dataOut,				// size 4*4 int2s per block
	__global const int* histSum,		// size 16  per block (64 B)
	__global const int* blockHists,		// size 16 int2s per block (64 B)
	const int bitOffset,				// k*4, k=0..7
	const int N)						// N = 32 (32x int2 global)
{
    const int4 gid4 = ((const int4)(get_global_id(0) << 2)) + (const int4)(0,1,2,3);
    const int tid = get_local_id(0);
    const int4 tid4 = ((int4)(tid << 2)) + (int4)(0,1,2,3);
    const int blockId = get_group_id(0);
    const int blockSize = get_local_size(0);
    const int numBlocks = get_num_groups(0);
    __local int sharedHistSum[16];
    __local int localHistStart[16];

    // Fetch per-block int2 histogram and int histogram sums
    if (tid < 16)
    {
        sharedHistSum[tid] = histSum[tid * numBlocks + blockId];
        localHistStart[tid] = blockHists[(blockId << 5) + tid];
    }

    // Copy data, each thread copies 4 (Cell,Tri) pairs into local shared mem
    barrier(CLK_LOCAL_MEM_FENCE);
    int2 myData[4];
    int myShiftedKeys[4];
    myData[0] = (gid4.x < N) ? dataIn[gid4.x] : MAX_INT2;
    myData[1] = (gid4.y < N) ? dataIn[gid4.y] : MAX_INT2;
    myData[2] = (gid4.z < N) ? dataIn[gid4.z] : MAX_INT2;
    myData[3] = (gid4.w < N) ? dataIn[gid4.w] : MAX_INT2;

    myShiftedKeys[0] = ((int)myData[0].x >> bitOffset) & 0xF;
    myShiftedKeys[1] = ((int)myData[1].x >> bitOffset) & 0xF;
    myShiftedKeys[2] = ((int)myData[2].x >> bitOffset) & 0xF;
    myShiftedKeys[3] = ((int)myData[3].x >> bitOffset) & 0xF;

	// Necessary ?
    //barrier(CLK_LOCAL_MEM_FENCE);

    // Compute the final indices
    int4 finalOffset;
    finalOffset.x = tid4.x - localHistStart[myShiftedKeys[0]] + sharedHistSum[myShiftedKeys[0]];
    finalOffset.y = tid4.y - localHistStart[myShiftedKeys[1]] + sharedHistSum[myShiftedKeys[1]];
    finalOffset.z = tid4.z - localHistStart[myShiftedKeys[2]] + sharedHistSum[myShiftedKeys[2]];
    finalOffset.w = tid4.w - localHistStart[myShiftedKeys[3]] + sharedHistSum[myShiftedKeys[3]];

    // Permute the data to the final offsets
    if (gid4.x < N) dataOut[finalOffset.x] = myData[0];
    if (gid4.y < N) dataOut[finalOffset.y] = myData[1];
    if (gid4.z < N) dataOut[finalOffset.z] = myData[2];
    if (gid4.w < N) dataOut[finalOffset.w] = myData[3];
}