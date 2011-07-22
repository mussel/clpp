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
// http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
//------------------------------------------------------------

// To do : visiting logic and multi-scan.

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

#ifdef KEYS_ONLY
#define KEY(DATA) (DATA)
#else
#define KEY(DATA) (DATA.x)
#endif

#define EXTRACT_KEY_BIT(VALUE,BIT) ((KEY(VALUE)>>BIT)&0x1)
#define EXTRACT_KEY_4BITS(VALUE,BIT) ((KEY(VALUE)>>BIT)&0xF)

#define BARRIER_LOCAL barrier(CLK_LOCAL_MEM_FENCE)

//------------------------------------------------------------
// exclusive_scan_128
//
// Purpose : Do a scan of 128 elements in once.
//------------------------------------------------------------

inline
void exclusive_scan_4(const uint tid, const int4 tid4, __local uint* localBuffer, __local uint* bitsOnCount)
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
        bitsOnCount[0] = localBuffer[tid4.w];
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
			
            uint tmp = localBuffer[ai];
            localBuffer[ai] = localBuffer[bi];
            localBuffer[bi] += tmp;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

inline 
void exclusive_scan_128(const uint tid, const int4 tid4, __local uint* localBuffer, __local uint* bitsOnCount)
{
	// local serial reduction
	localBuffer[tid4.y] += localBuffer[tid4.x];
	localBuffer[tid4.w] += localBuffer[tid4.z];
	localBuffer[tid4.w] += localBuffer[tid4.y];
		
	// Exclusive scan starting with an offset of 4
	exclusive_scan_4(tid, tid4, localBuffer, bitsOnCount);
		
	// local 4-element serial expansion
	uint tmp;
	tmp = localBuffer[tid4.y];    localBuffer[tid4.y] = localBuffer[tid4.w];  localBuffer[tid4.w] += tmp;
	tmp = localBuffer[tid4.x];    localBuffer[tid4.x] = localBuffer[tid4.y];  localBuffer[tid4.y] += tmp;
	tmp = localBuffer[tid4.z];    localBuffer[tid4.z] = localBuffer[tid4.w];  localBuffer[tid4.w] += tmp;
}

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
	__global KV_TYPE* data,				// size 4*4 int2s per block (8 kB)
	const int bitOffset,				// k*4, k=0..7
	const int N)						// Total number of items to sort
{
	const int tid = (int)get_local_id(0);
		
    const int4 gid4 = (int4)(get_global_id(0) << 2) + (const int4)(0,1,2,3);    
    const int4 tid4 = (int4)(tid << 2) + (const int4)(0,1,2,3);
    
	// Local memory
	__local uint localBitsScan[WGZ_x4];
    __local uint bitsOnCount[1];

    // Each thread copies 4 (Cell,Tri) pairs into local memory
    localData[tid4.x] = (gid4.x < N) ? data[gid4.x] : MAX_KV_TYPE;
    localData[tid4.y] = (gid4.y < N) ? data[gid4.y] : MAX_KV_TYPE;
    localData[tid4.z] = (gid4.z < N) ? data[gid4.z] : MAX_KV_TYPE;
    localData[tid4.w] = (gid4.w < N) ? data[gid4.w] : MAX_KV_TYPE;
	
	//-------- 1) 4 x local 1-bit split

	__local KV_TYPE* localTemp = localData + WGZ_x4;
	#pragma unroll // SLOWER on some cards!!
    for(uint shift = bitOffset; shift < (bitOffset+4); shift++) // Radix 4
    {
		BARRIER_LOCAL;
		
		//---- Setup the array of 4 bits (of level shift)
		// Create the '1s' array as explained at : http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
		// In fact we simply inverse the bits	
		// Local copy and bits extraction
		int4 flags;
		flags.x = localBitsScan[tid4.x] = ! EXTRACT_KEY_BIT(localData[tid4.x], shift);
        flags.y = localBitsScan[tid4.y] = ! EXTRACT_KEY_BIT(localData[tid4.y], shift);
        flags.z = localBitsScan[tid4.z] = ! EXTRACT_KEY_BIT(localData[tid4.z], shift);
        flags.w = localBitsScan[tid4.w] = ! EXTRACT_KEY_BIT(localData[tid4.w], shift);
						
		//---- Do a scan of the 128 bits and retreive the total number of '1' in 'bitsOnCount'
		exclusive_scan_128(tid, tid4, localBitsScan, bitsOnCount);
		
		BARRIER_LOCAL;
		
		//----
		int offset;
		int4 invFlags = 1 - flags;
		
		offset = invFlags.x * (bitsOnCount[0] + tid4.x - localBitsScan[tid4.x]) + flags.x * localBitsScan[tid4.x];
		localTemp[offset] = localData[tid4.x];
		
		offset = invFlags.y * (bitsOnCount[0] + tid4.y - localBitsScan[tid4.y]) + flags.y * localBitsScan[tid4.y];
		localTemp[offset] = localData[tid4.y];
		
		offset = invFlags.z * (bitsOnCount[0] + tid4.z - localBitsScan[tid4.z]) + flags.z * localBitsScan[tid4.z];
		localTemp[offset] = localData[tid4.z];
				
		offset = invFlags.w * (bitsOnCount[0] + tid4.w - localBitsScan[tid4.w]) + flags.w * localBitsScan[tid4.w];
		localTemp[offset] = localData[tid4.w];
		
		BARRIER_LOCAL;

		// Swap the buffer pointers
		__local KV_TYPE* swBuf = localData;
		localData = localTemp;
		localTemp = swBuf;
		
		//barrier(CLK_LOCAL_MEM_FENCE); // NO CRASH !!			
    }
	
	// FASTER !!
	//barrier(CLK_LOCAL_MEM_FENCE); // NO CRASH !!
	
	// Write sorted data back to global memory
	if (gid4.x < N) data[gid4.x] = localData[tid4.x];
    if (gid4.y < N) data[gid4.y] = localData[tid4.y];
    if (gid4.z < N) data[gid4.z] = localData[tid4.z];
    if (gid4.w < N) data[gid4.w] = localData[tid4.w];	
}

//------------------------------------------------------------
// kernel__localHistogram
//
// Purpose :
//
// Given an array of 'locally sorted' blocks of keys (according to a 4-bit radix), for each 
// block we counts the number of keys that fall into each radix, and finds the starting
// offset of each radix in the block.
//
// It then writes the radix counts to the 'radixCount' array, and the starting offsets to the 'radixOffsets' array.
//------------------------------------------------------------

__kernel
void kernel__localHistogram(__global KV_TYPE* data, const int bitOffset, __global uint* radixCount, __global uint* radixOffsets, const int N)
{
    const int tid = (int)get_local_id(0);
    const int4 tid4 = (int4)(tid << 2) + (const int4)(0,1,2,3);
	const int4 gid4 = (int4)(get_global_id(0) << 2) + (const int4)(0,1,2,3);
	const int blockId = (int)get_group_id(0);
	
	__local uint localData[WGZ_x4];
    __local int localHistStart[16];
    __local int localHistEnd[16];
	
	//---- Extract the radix
    localData[tid4.x] = (gid4.x < N) ? EXTRACT_KEY_4BITS(data[gid4.x], bitOffset) : EXTRACT_KEY_4BITS(MAX_KV_TYPE, bitOffset);
    localData[tid4.y] = (gid4.y < N) ? EXTRACT_KEY_4BITS(data[gid4.y], bitOffset) : EXTRACT_KEY_4BITS(MAX_KV_TYPE, bitOffset);
    localData[tid4.z] = (gid4.z < N) ? EXTRACT_KEY_4BITS(data[gid4.z], bitOffset) : EXTRACT_KEY_4BITS(MAX_KV_TYPE, bitOffset);
    localData[tid4.w] = (gid4.w < N) ? EXTRACT_KEY_4BITS(data[gid4.w], bitOffset) : EXTRACT_KEY_4BITS(MAX_KV_TYPE, bitOffset);
	
	//---- Create the histogram

    BARRIER_LOCAL;
	
	// Reset the local histogram
    if (tid < 16)
    {
        localHistStart[tid] = 0;
        localHistEnd[tid] = -1;
    }
	BARRIER_LOCAL;
	
    // Finds the position where the localData entries differ and stores start index (localHistStart) for each radix.
	// This way, for the first 'instance' of a radix, we store its index.
	// We also store where each radix ends in 'localHistEnd'.
	//
	// And so, if we use end-start+1 we have the histogram value to store.
	
    if (tid4.x > 0 && localData[tid4.x] != localData[tid4.x-1])
    {
		localHistStart[localData[tid4.x]] = tid4.x;
        localHistEnd[localData[tid4.x-1]] = tid4.x - 1;        
    }

    if (localData[tid4.y] != localData[tid4.x])
    {
		localHistStart[localData[tid4.y]] = tid4.y;
        localHistEnd[localData[tid4.x]] = tid4.x;        
    }

    if (localData[tid4.z] != localData[tid4.y])
    {
		localHistStart[localData[tid4.z]] = tid4.z;
        localHistEnd[localData[tid4.y]] = tid4.y;        
    }

    if (localData[tid4.w] != localData[tid4.z])
    {
		localHistStart[localData[tid4.w]] = tid4.w;
        localHistEnd[localData[tid4.z]] = tid4.z;
    }

	// First and last histogram values
    if (tid < 1)
    {
		localHistStart[localData[0]] = 0;
		localHistEnd[localData[WGZ_x4-1]] = WGZ_x4 - 1;		
    }
    BARRIER_LOCAL;

    //---- Write the 16 histogram values back to the global memory
    if (tid < 16)
    {
        radixCount[tid * get_num_groups(0) + blockId] = localHistEnd[tid] - localHistStart[tid] + 1;
		radixOffsets[(blockId << 4) + tid] = localHistStart[tid];
    }
}

//------------------------------------------------------------
// kernel__radixPermute
//
// Purpose : Prefix sum results are used to scatter each work-group's elements to their correct position.
//------------------------------------------------------------

__kernel
void kernel__radixPermute(
	__global const KV_TYPE* dataIn,
	__global KV_TYPE* dataOut,
	__global const int* histSum,
	__global const int* radixOffsets,
	const uint bitOffset,
	const uint N,
	const int numBlocks)
{
    const int4 gid4 = ((const int4)(get_global_id(0) << 2)) + (const int4)(0,1,2,3);
    const int tid = get_local_id(0);
    const int4 tid4 = ((const int4)(tid << 2)) + (const int4)(0,1,2,3);
    
    //const int numBlocks = get_num_groups(0); // Can be passed as a parameter !
    __local int sharedHistSum[16];
    __local int localHistStart[16];

    // Fetch per-block KV_TYPE histogram and int histogram sums
    if (tid < 16)
    {
		const uint blockId = get_group_id(0);
        sharedHistSum[tid] = histSum[tid * numBlocks + blockId];
        localHistStart[tid] = radixOffsets[(blockId << 4) + tid];
    }
	
	BARRIER_LOCAL;

    // Retreive the data in local memory for faster access
    KV_TYPE myData[4];
    uint myShiftedKeys[4];
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
    uint4 finalOffset;
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