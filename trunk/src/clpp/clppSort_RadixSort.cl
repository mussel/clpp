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

#pragma OPENCL EXTENSION cl_amd_printf : enable

#define WGZ 32
#define WGZ_x2 (WGZ*2)
#define WGZ_x3 (WGZ*3)
#define WGZ_x4 (WGZ*4)
#define WGZ_1 (WGZ-1)
#define WGZ_2 (WGZ-2)
#define WGZ_x2_1 (WGZ_x2-1)
#define WGZ_x3_1 (WGZ_x3-1)
#define WGZ_x4_1 (WGZ_x4-1)

//#define EXTRACT_KEY_BIT(VALUE,BIT) ((((uint)VALUE)>>BIT)&0x1)
//#define EXTRACT_KEY_4BITS(VALUE,BIT) ((((uint)VALUE)>>BIT)&0x0F)

#if KEYS_ONLY
#define KEY(DATA) (DATA)
#else
#define KEY(DATA) (DATA.x)
#endif

#define EXTRACT_KEY_BIT(VALUE,BIT) ((((uint)KEY(VALUE))>>(uint)BIT)&0x1)
#define EXTRACT_KEY_4BITS(VALUE,BIT) ((((uint)KEY(VALUE))>>(uint)BIT)&0xF)

#if defined(OCL_DEVICE_GPU) && defined(OCL_PLATFORM_NVIDIA)

// Because our workgroup size = SIMT size, we use the natural synchronization provided by SIMT.
// So, we don't need any barrier to synchronize
#define BARRIER_LOCAL barrier(CLK_GLOBAL_MEM_FENCE)

// Sometimes using barrier give a faster sort... but it is random !!!
//#define BARRIER_LOCAL barrier(CLK_LOCAL_MEM_FENCE)

#else

#define BARRIER_LOCAL barrier(CLK_LOCAL_MEM_FENCE)

#endif

#if defined(OCL_DEVICE_GPU) && defined(OCL_PLATFORM_NVIDIA)

// Inclusive scan of 4 buckets of 32 elements by using the SIMT capability (to avoid synchronization of work items).
// Directly do it for 4x32 elements, simply use an offset
inline void scan_simt_inclusive_4(__local K_TYPE* input, const int tid1)
{
	const uint tid2 = tid1 + 32;
	const uint tid3 = tid2 + 32;
	const uint tid4 = tid3 + 32;
	
	if (tid1 > 0 )
	{
		input[tid1] += input[tid1 - 1];
		input[tid2] += input[tid2 - 1];
		input[tid3] += input[tid3 - 1];
		input[tid4] += input[tid4 - 1];
	}
	
	if (tid1 > 1 )
	{
		input[tid1] += input[tid1 - 2];
		input[tid2] += input[tid2 - 2];
		input[tid3] += input[tid3 - 2];
		input[tid4] += input[tid4 - 2];
	}
	
	if (tid1 > 3 )
	{
		input[tid1] += input[tid1 - 4];
		input[tid2] += input[tid2 - 4];
		input[tid3] += input[tid3 - 4];
		input[tid4] += input[tid4 - 4];
	}
	
	if (tid1 > 7 )
	{
		input[tid1] += input[tid1 - 8];
		input[tid2] += input[tid2 - 8];
		input[tid3] += input[tid3 - 8];
		input[tid4] += input[tid4 - 8];
	}
	
	if (tid1 > 15)
	{
		input[tid1] += input[tid1 - 16];
		input[tid2] += input[tid2 - 16];
		input[tid3] += input[tid3 - 16];
		input[tid4] += input[tid4 - 16];
	}
}

inline 
void exclusive_scan_128(const uint tid, const int4 tid4, __local K_TYPE* localBuffer, __local K_TYPE* incSum, uint size)
{
	// Do 4 inclusive scan (4 buckets of 32)
	scan_simt_inclusive_4(localBuffer, tid);
	
	BARRIER_LOCAL;
		
	// Convert into a scan of 128 items
	__local int sum[3];
	if (tid > WGZ_2)
	{
		sum[0] = localBuffer[WGZ_1];
		sum[1] = sum[0] + localBuffer[WGZ_x2_1];
		sum[2] = sum[1] + localBuffer[WGZ_x3_1];
	}       
		
	BARRIER_LOCAL;
			
	// Add the sum to the other buckets
	localBuffer[tid + WGZ]		+= sum[0];
	localBuffer[tid + WGZ_x2]	+= sum[1];
	localBuffer[tid + WGZ_x3]	+= sum[2];
	
	// Total number of '1' in the array, retreived from the inclusive scan
	if (tid > WGZ_2)
		incSum[0] = localBuffer[WGZ_x4_1];
	
	BARRIER_LOCAL;
	
	// To exclusive scan
	K_TYPE v1 = (tid > 0) ? localBuffer[tid4.x - 1] : K_TYPE_IDENTITY;
	K_TYPE v2 = localBuffer[tid4.y - 1];
	K_TYPE v3 = localBuffer[tid4.z - 1];
	K_TYPE v4 = localBuffer[tid4.w - 1];

	localBuffer[tid4.x] = v1;
	localBuffer[tid4.y] = v2;
	localBuffer[tid4.z] = v3;
	localBuffer[tid4.w] = v4;
		
	BARRIER_LOCAL;
}

#else

inline
void exclusive_scan_4(const uint tid, const int4 tid4, __local K_TYPE* localBuffer, __local K_TYPE* incSum)
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
    if (tid > WGZ_2)
    {
        incSum[0] = localBuffer[tid4.w];
        localBuffer[tid4.w] = 0;
    }

    // expansion
	//#pragma unroll
    for (uint d = 1; d < 32; d <<= 1)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
		offset >>= 1;
		
        if (tid < d)
        {
            const uint ai = mad24(offset, (tid2_1+0), -1); // offset*(tid2_0+1)-1 = offset*(tid2_1+0)-1
            const uint bi = mad24(offset, (tid2_1+1), -1); // offset*(tid2_1+1)-1;
			
            K_TYPE tmp = localBuffer[ai];
            localBuffer[ai] = localBuffer[bi];
            localBuffer[bi] += tmp;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

inline 
void exclusive_scan_128(const uint tid, const int4 tid4, __local K_TYPE* localBuffer, __local K_TYPE* incSum, uint size)
{
	// local serial reduction
	localBuffer[tid4.y] += localBuffer[tid4.x];
	localBuffer[tid4.w] += localBuffer[tid4.z];
	localBuffer[tid4.w] += localBuffer[tid4.y];
		
	// Exclusive scan starting with an offset of 4
	exclusive_scan_4(tid, tid4, localBuffer, incSum);
		
	// local 4-element serial expansion
	K_TYPE tmp;
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
	__local KV_TYPE* localData,			// size 4*4 int2s (8 kB)
	__local K_TYPE* indices,			// size 4*4 shorts (4 kB)
	//__local K_TYPE* localBitsScanOLD,	// size 4*4*2 shorts (2 kB)
	__global KV_TYPE* data,				// size 4*4 int2s per block (8 kB)
	__global int* hist,					// size 16  per block (64 B)
	__global int* blockHists,			// size 16 int2s per block (64 B)
	const int bitOffset,				// k*4, k=0..7
	const int N)						// Total number of items to sort
{
    const int4 gid4 = (int4)(get_global_id(0) << 2) + (const int4)(0,1,2,3);
    const int tid = (int)get_local_id(0);
    const int4 tid4 = (int4)(tid << 2) + (const int4)(0,1,2,3);
    const int blockId = (int)get_group_id(0);
	
	// DOES NOT WORK !!!!
	//__local KV_TYPE localData[WGZ_x4 * 2]; // 2 KV array of 128 items (2 for permutations)
	__local K_TYPE localBitsScan[WGZ_x4];

    __local int localHistStart[16];
    __local int localHistEnd[16];
    __local K_TYPE incSum[1];

    // Each thread copies 4 (Cell,Tri) pairs into local memory
    localData[tid4.x] = (gid4.x < N) ? data[gid4.x] : MAX_KV_TYPE;
    localData[tid4.y] = (gid4.y < N) ? data[gid4.y] : MAX_KV_TYPE;
    localData[tid4.z] = (gid4.z < N) ? data[gid4.z] : MAX_KV_TYPE;
    localData[tid4.w] = (gid4.w < N) ? data[gid4.w] : MAX_KV_TYPE;
	
    indices[tid4.x] = tid4.x;
    indices[tid4.y] = tid4.y;
    indices[tid4.z] = tid4.z;
    indices[tid4.w] = tid4.w;

    int srcBase = 0;
    int dstBase = WGZ_x4;
	
	//-------- 1) 4 x local 1-bit split

    uint shift = bitOffset;
	//#pragma unroll // SLOWER !!
    for(uint i = 0; i < 4; i++, shift++)
    {
		//---- Setup the array of 4 bits (of level shift)
		// Create the '1s' array as explained at : http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
		// In fact we simply inverse the bits
        BARRIER_LOCAL;
		
        localBitsScan[tid4.x] = ! EXTRACT_KEY_BIT(localData[indices[srcBase + tid4.x]], shift);
        localBitsScan[tid4.y] = ! EXTRACT_KEY_BIT(localData[indices[srcBase + tid4.y]], shift);
        localBitsScan[tid4.z] = ! EXTRACT_KEY_BIT(localData[indices[srcBase + tid4.z]], shift);
        localBitsScan[tid4.w] = ! EXTRACT_KEY_BIT(localData[indices[srcBase + tid4.w]], shift);
		
		//--- Do a scan of the 128 bits and retreive the total number of '1' in 'incSum'
		exclusive_scan_128(tid, tid4, localBitsScan, incSum, N);

        //---- Permutations
		#pragma unroll
        for(uint b = 0; b < 4; b++)
        {
            uint idx = tid4.x + b;
            BARRIER_LOCAL;
			
			uint flag = EXTRACT_KEY_BIT(localData[indices[srcBase + idx]], shift);
			
			// Rule :
			// t = idx - scan[idx] + total_count_bits_at0
			// pos = flag ? t : scan[idx]
			
            //if (flag == 1)
            //    indices[dstBase + (int)incSum[0] + idx - (int)localBitsScan[idx]] = indices[srcBase + idx];
            //else
            //    indices[dstBase + (int)localBitsScan[idx]] = indices[srcBase + idx];
				
			// Faster version for GPU (no divergence)
			uint targetOffset = flag * ( (int)incSum[0] + idx - (int)localBitsScan[idx] ) + (1-flag) * ((int)localBitsScan[idx]);
			indices[dstBase + targetOffset] = indices[srcBase + idx];
        }

        // Pingpong left and right halves of the indirection buffer
        int tmpBase = srcBase;
		srcBase = dstBase;
		dstBase = tmpBase;
    }
	
    BARRIER_LOCAL;
	
	// Write sorted data back to global mem
    if (gid4.x < N) data[gid4.x] = localData[indices[srcBase + tid4.x]];
    if (gid4.y < N) data[gid4.y] = localData[indices[srcBase + tid4.y]];
    if (gid4.z < N) data[gid4.z] = localData[indices[srcBase + tid4.z]];
    if (gid4.w < N) data[gid4.w] = localData[indices[srcBase + tid4.w]];
	
	//-------- 2) Histogram

    // init histogram values
    BARRIER_LOCAL;
    if (tid < 16)
    {
        localHistStart[tid] = 0;
        localHistEnd[tid] = -1;
    }

    // Start computation
	
    BARRIER_LOCAL;
    int ka, kb;
    if (tid4.x > 0)
    {
        ka = EXTRACT_KEY_4BITS(localData[indices[srcBase + tid4.x-0]], bitOffset);
        kb = EXTRACT_KEY_4BITS(localData[indices[srcBase + tid4.x-1]], bitOffset);
        if (ka != kb)
        {
            localHistEnd[kb] = tid4.x - 1;
            localHistStart[ka] = tid4.x;
        }
    }

    ka = EXTRACT_KEY_4BITS(localData[indices[srcBase + tid4.y]], bitOffset);
    kb = EXTRACT_KEY_4BITS(localData[indices[srcBase + tid4.x]], bitOffset);
    if (ka != kb)
    {
        localHistEnd[kb] = tid4.x;
        localHistStart[ka] = tid4.y;
    }

    ka = EXTRACT_KEY_4BITS(localData[indices[srcBase + tid4.z]], bitOffset);
    kb = EXTRACT_KEY_4BITS(localData[indices[srcBase + tid4.y]], bitOffset);
    if (ka != kb)
    {
        localHistEnd[kb] = tid4.y;
        localHistStart[ka] = tid4.z;
    }

    ka = EXTRACT_KEY_4BITS(localData[indices[srcBase + tid4.w]], bitOffset);
    kb = EXTRACT_KEY_4BITS(localData[indices[srcBase + tid4.z]], bitOffset);
    if (ka != kb)
    {
        localHistEnd[kb] = tid4.z;
        localHistStart[ka] = tid4.w;
    }

    if (tid < 1)
    {
		localHistEnd[EXTRACT_KEY_4BITS(localData[indices[srcBase + WGZ_x4-1]], bitOffset)] = WGZ_x4 - 1;
		localHistStart[EXTRACT_KEY_4BITS(localData[indices[srcBase]], bitOffset)] = 0;
    }
    BARRIER_LOCAL;

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
	__global const KV_TYPE* dataIn,		// size 4*4 int2s per block
	__global KV_TYPE* dataOut,				// size 4*4 int2s per block
	__global const int* histSum,		// size 16 per block (64 B)
	__global const int* blockHists,		// size 16 int2s per block (64 B)
	const int bitOffset,				// k*4, k=0..7
	const int N)						// N = 32 (32x KV_TYPE global)
{
    const int4 gid4 = ((const int4)(get_global_id(0) << 2)) + (const int4)(0,1,2,3);
    const int tid = get_local_id(0);
    const int4 tid4 = ((int4)(tid << 2)) + (int4)(0,1,2,3);
    const int blockId = get_group_id(0);
    const int numBlocks = get_num_groups(0);
    __local int sharedHistSum[16];
    __local int localHistStart[16];

    // Fetch per-block KV_TYPE histogram and int histogram sums
    if (tid < 16)
    {
        sharedHistSum[tid] = histSum[tid * numBlocks + blockId];
        localHistStart[tid] = blockHists[(blockId << 5) + tid];
    }
	
	BARRIER_LOCAL;

    // Copy data, each thread copies 4 (Cell,Tri) pairs into local memory
    KV_TYPE myData[4];
    int myShiftedKeys[4];
    myData[0] = (gid4.x < N) ? dataIn[gid4.x] : MAX_KV_TYPE;
    myData[1] = (gid4.y < N) ? dataIn[gid4.y] : MAX_KV_TYPE;
    myData[2] = (gid4.z < N) ? dataIn[gid4.z] : MAX_KV_TYPE;
    myData[3] = (gid4.w < N) ? dataIn[gid4.w] : MAX_KV_TYPE;

    myShiftedKeys[0] = EXTRACT_KEY_4BITS(myData[0], bitOffset);
    myShiftedKeys[1] = EXTRACT_KEY_4BITS(myData[1], bitOffset);
    myShiftedKeys[2] = EXTRACT_KEY_4BITS(myData[2], bitOffset);
    myShiftedKeys[3] = EXTRACT_KEY_4BITS(myData[3], bitOffset);

	// Necessary ?
    //BARRIER_LOCAL;

    // Compute the final indices
    int4 finalOffset;
    finalOffset.x = tid4.x - localHistStart[myShiftedKeys[0]] + sharedHistSum[myShiftedKeys[0]];
    finalOffset.y = tid4.y - localHistStart[myShiftedKeys[1]] + sharedHistSum[myShiftedKeys[1]];
    finalOffset.z = tid4.z - localHistStart[myShiftedKeys[2]] + sharedHistSum[myShiftedKeys[2]];
    finalOffset.w = tid4.w - localHistStart[myShiftedKeys[3]] + sharedHistSum[myShiftedKeys[3]];

    // Permute the data to the final offsets
	if (finalOffset.x < N) dataOut[finalOffset.x] = myData[0];
    if (finalOffset.y < N) dataOut[finalOffset.y] = myData[1];
    if (finalOffset.z < N) dataOut[finalOffset.z] = myData[2];
    if (finalOffset.w < N) dataOut[finalOffset.w] = myData[3];
}