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

//#define EXTRACT_KEY_BIT(VALUE,BIT) ((((uint)KEY(VALUE))>>(uint)BIT)&0x1)
//#define EXTRACT_KEY_4BITS(VALUE,BIT) ((((uint)KEY(VALUE))>>(uint)BIT)&0xF)
#define EXTRACT_KEY_BIT(VALUE,BIT) ((KEY(VALUE)>>BIT)&0x1)
#define EXTRACT_KEY_4BITS(VALUE,BIT) ((KEY(VALUE)>>BIT)&0xF)

#if defined(OCL_DEVICE_GPU) && defined(OCL_PLATFORM_NVIDIA)

// Because our workgroup size = SIMT size, we use the natural synchronization provided by SIMT.
// So, we don't need any barrier to synchronize
#define BARRIER_LOCAL

#else

#define BARRIER_LOCAL barrier(CLK_LOCAL_MEM_FENCE)

#endif

#if defined(OCL_DEVICE_GPU) && defined(OCL_PLATFORM_NVIDIA)
/*
#define SIMT_SIZE 32
#define SIMT_SIZE_1 (SIMT_SIZE-1)
#define SIMT_SIZE_2 (SIMT_SIZE-2)

uint scan_simt(uint val, volatile __local uint* sData, int maxlevel)
{
    // The following is the same as 2 * WARP_SIZE * warpId + threadInWarp = 64*(threadIdx.x >> 5) + (threadIdx.x & (WARP_SIZE - 1))
    int tid = get_local_id(0);
    int idx = 2 * tid - (tid & SIMT_SIZE_1);
	
    sData[idx] = 0;
    idx += SIMT_SIZE;
    sData[idx] = val;     

    if (0 <= maxlevel) { sData[idx] += sData[idx - 1]; }
    if (1 <= maxlevel) { sData[idx] += sData[idx - 2]; }
    if (2 <= maxlevel) { sData[idx] += sData[idx - 4]; }
    if (3 <= maxlevel) { sData[idx] += sData[idx - 8]; }
    if (4 <= maxlevel) { sData[idx] += sData[idx -16]; }

    return sData[idx] - val;  // convert inclusive -> exclusive
}

inline
uint4 scan4(uint4 idata, __local uint* ptr)
{        
    uint idx = get_local_id(0);
	const uint lane = idx & SIMT_SIZE_1;
	const uint wiBlockId = idx >> 5;

	// Scan the int4 and store it in 'sum'
    uint4 val4 = idata;
    uint sum[3];
    sum[0] = val4.x;
    sum[1] = val4.y + sum[0];
    sum[2] = val4.z + sum[1];    
    uint val = val4.w + sum[2];
    
	// Scan the warp
    val = scan_simt(val, ptr, 4);
    BARRIER_LOCAL;
	
    if (lane > SIMT_SIZE_2)
        ptr[wiBlockId] = val + val4.w + sum[2];
		
    BARRIER_LOCAL;

	if (idx < SIMT_SIZE)
		ptr[idx] = scan_simt(ptr[idx], ptr, 2);
    
    BARRIER_LOCAL;

    val += ptr[wiBlockId];

    val4.x = val;
    val4.y = val + sum[0];
    val4.z = val + sum[1];
    val4.w = val + sum[2];

    return val4;
}

void exclusive_scan_128(const uint tid, const int4 tid4, __local uint* localBuffer, __local uint* bitsOnCount)
{
	int localSize = get_local_size(0);
	
	uint4 preds;
	preds.x = localBuffer[tid4.x];
	preds.y = localBuffer[tid4.y];
	preds.z = localBuffer[tid4.z];
	preds.w = localBuffer[tid4.w];

	uint4 address = scan4(preds, localBuffer);
	
	localBuffer[tid4.x] = address.x;
	localBuffer[tid4.y] = address.y;
	localBuffer[tid4.z] = address.z;
	localBuffer[tid4.w] = address.w;
	
	if (tid == localSize - 1) 
		bitsOnCount[0] = address.w + preds.w; // reconvert to inclusive
	
	BARRIER_LOCAL;
}
*/
/*
// Inclusive scan of 4 buckets of 32 elements by using the SIMT capability (to avoid synchronization of work items).
// Directly do it for 4x32 elements, simply use an offset
inline void scan_simt_inclusive_4(__local uint* input, const int tid1)
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
void exclusive_scan_128(const uint tid, const int4 tid4, __local uint* localBuffer, __local uint* bitsOnCount)
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
		bitsOnCount[0] = localBuffer[WGZ_x4_1];
	
	BARRIER_LOCAL;
	
	// To exclusive scan
	uint v1 = (tid > 0) ? localBuffer[tid4.x - 1] : K_TYPE_IDENTITY;
	uint v2 = localBuffer[tid4.y - 1];
	uint v3 = localBuffer[tid4.z - 1];
	uint v4 = localBuffer[tid4.w - 1];

	localBuffer[tid4.x] = v1;
	localBuffer[tid4.y] = v2;
	localBuffer[tid4.z] = v3;
	localBuffer[tid4.w] = v4;
		
	BARRIER_LOCAL;
}
*/

inline 
void exclusive_scan_128(const uint tid, int4 tid4, __local uint* localBuffer, __local uint* bitsOnCount)
{
	// We don't use the same tid4 !
	tid4 = (int4)(tid << 2) + (const int4)(0,1,2,3);
	
	localBuffer[tid4.y] += localBuffer[tid4.x];
	localBuffer[tid4.z] += localBuffer[tid4.y];
	localBuffer[tid4.w] += localBuffer[tid4.z];
	
	// Inclusive scan
	if (tid > 0 )	localBuffer[tid4.w] += localBuffer[tid4.w - 1*4];
	if (tid > 1 )	localBuffer[tid4.w] += localBuffer[tid4.w - 2*4];
	if (tid > 3 )	localBuffer[tid4.w] += localBuffer[tid4.w - 4*4];
	if (tid > 7 )	localBuffer[tid4.w] += localBuffer[tid4.w - 8*4];
	if (tid > 15)	localBuffer[tid4.w] += localBuffer[tid4.w - 16*4];
	
	// Total number of '1' in the array, retreived from the inclusive scan
	if (tid > WGZ_2)
		bitsOnCount[0] = localBuffer[WGZ_x4_1];
	
	// 1 - To exclusive scan
	// 2 - Add the sums
	int toAdd = (tid > 0) ? localBuffer[tid4.x-1] : 0;
		
	localBuffer[tid4.w] = localBuffer[tid4.z] + toAdd;
	localBuffer[tid4.z] = localBuffer[tid4.y] + toAdd;
	localBuffer[tid4.y] = localBuffer[tid4.x] + toAdd;
	localBuffer[tid4.x] = toAdd;
}
#else

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
			
            K_TYPE tmp = localBuffer[ai];
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
	__global KV_TYPE* data,				// size 4*4 int2s per block (8 kB)
	const int bitOffset,				// k*4, k=0..7
	const int N)						// Total number of items to sort
{
	const int tid = (int)get_local_id(0);
		
#if defined(OCL_DEVICE_GPU) && defined(OCL_PLATFORM_NVIDIA)
	const int groupId = get_group_id(0);
    const int4 tid4 = ((const int4)tid) + (const int4)(0,WGZ,WGZ_x2,WGZ_x3);		
	const int4 gid4 = tid4 + ((const int4)groupId<<2);
#else
    const int4 gid4 = (int4)(get_global_id(0) << 2) + (const int4)(0,1,2,3);    
    const int4 tid4 = (int4)(tid << 2) + (const int4)(0,1,2,3);
	#define lid4 tid4
#endif
    
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
		
		//---- 
		
        /*for(uint b = 0; b < 4; b++)
        {
            uint idx = tid4.x + b;
            barrier(CLK_LOCAL_MEM_FENCE);
			
            int flag = EXTRACT_KEY_BIT(localData[idx], shift);
			
            //if (flag == 1)
            //    indices[dstBase + (int)bitsOnCount[0] + idx - (int)localBitsScan[idx]] = indices[srcBase + idx];
            //else
            //    indices[dstBase + (int)localBitsScan[idx]] = indices[srcBase + idx];
				
			// Faster version for GPU (no divergence)
			int targetOffset = flag * ((int)bitsOnCount[0] + idx - (int)localBitsScan[idx]) + (1-flag) * ((int)localBitsScan[idx]);
			localTemp[targetOffset] = localData[idx];
        }
		
		// Swap the buffer pointers
		__local KV_TYPE* swBuf = localData;
		localData = localTemp;
		localTemp = swBuf;*/
		
		//const int4 lid4 = (int4)(tid << 2) + (const int4)(0,1,2,3);
				
		
		//---- Compute the rank
		/*uint4 r;
		r.x = (preds.x) ? localBitsScan[tid4.x] : bitsOnCount[0] + tid4.x - localBitsScan[tid4.x];
		r.y = (preds.y) ? localBitsScan[tid4.y] : bitsOnCount[0] + tid4.y - localBitsScan[tid4.y];
		r.z = (preds.z) ? localBitsScan[tid4.z] : bitsOnCount[0] + tid4.z - localBitsScan[tid4.z];
		r.w = (preds.w) ? localBitsScan[tid4.w] : bitsOnCount[0] + tid4.w - localBitsScan[tid4.w];
		
		BARRIER_LOCAL;
		
		//---- Permute to the other half of the array (The array has 256 KV pairs).
		localTemp[(r.x & 3) * WGZ + (r.x >> 2)] = localData[tid4.x];
        localTemp[(r.y & 3) * WGZ + (r.y >> 2)] = localData[tid4.y];
        localTemp[(r.z & 3) * WGZ + (r.z >> 2)] = localData[tid4.z];
        localTemp[(r.w & 3) * WGZ + (r.w >> 2)] = localData[tid4.w];
		
		BARRIER_LOCAL;
		
		// The above allows us to read without 4-way bank conflicts
        localData[tid4.x] = localTemp[tid4.x];
        localData[tid4.y] = localTemp[tid4.y];
        localData[tid4.z] = localTemp[tid4.z];
        localData[tid4.w] = localTemp[tid4.w];*/
		
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
//------------------------------------------------------------

__kernel
void kernel__localHistogram(__global KV_TYPE* data, const int bitOffset, __global int* hist, __global int* blockHists, const int N)
{
    const int tid = (int)get_local_id(0);
    const int4 tid4 = (int4)(tid << 2) + (const int4)(0,1,2,3);
	const int4 gid4 = (int4)(get_global_id(0) << 2) + (const int4)(0,1,2,3);
	const int blockId = (int)get_group_id(0);
	
	__local KV_TYPE localData[WGZ*4];
    __local int localHistStart[16];
    __local int localHistEnd[16];
	
    localData[tid4.x] = (gid4.x < N) ? EXTRACT_KEY_4BITS(data[gid4.x], bitOffset) : EXTRACT_KEY_4BITS(MAX_KV_TYPE, bitOffset);
    localData[tid4.y] = (gid4.y < N) ? EXTRACT_KEY_4BITS(data[gid4.y], bitOffset) : EXTRACT_KEY_4BITS(MAX_KV_TYPE, bitOffset);
    localData[tid4.z] = (gid4.z < N) ? EXTRACT_KEY_4BITS(data[gid4.z], bitOffset) : EXTRACT_KEY_4BITS(MAX_KV_TYPE, bitOffset);
    localData[tid4.w] = (gid4.w < N) ? EXTRACT_KEY_4BITS(data[gid4.w], bitOffset) : EXTRACT_KEY_4BITS(MAX_KV_TYPE, bitOffset);
	
	//-------- 2) Histogram

    // init histogram values
    BARRIER_LOCAL;
    if (tid < 16)
    {
        localHistStart[tid] = 0;
        localHistEnd[tid] = -1;
    }
	BARRIER_LOCAL;

    // Start computation    
    if (tid4.x > 0 && localData[tid4.x] != localData[tid4.x-1])
    {
		localHistStart[localData[tid4.x]] = tid4.x;
        localHistEnd[localData[tid4.x-1]] = tid4.x - 1;        
    }

    if (localData[tid4.y] != localData[tid4.x])
    {
        localHistEnd[localData[tid4.x]] = tid4.x;
        localHistStart[localData[tid4.y]] = tid4.y;
    }

    if (localData[tid4.z] != localData[tid4.y])
    {
        localHistEnd[localData[tid4.y]] = tid4.y;
        localHistStart[localData[tid4.z]] = tid4.z;
    }

    if (localData[tid4.w] != localData[tid4.z])
    {
        localHistEnd[localData[tid4.z]] = tid4.z;
        localHistStart[localData[tid4.w]] = tid4.w;
    }

    if (tid < 1)
    {
		localHistEnd[localData[WGZ_x4-1]] = WGZ_x4 - 1;
		localHistStart[localData[0]] = 0;
    }
    BARRIER_LOCAL;

    // Write histogram to global memomry
    if (tid < 16)
    {
        hist[tid * get_num_groups(0) + blockId] = localHistEnd[tid] - localHistStart[tid] + 1;
		blockHists[(blockId << 5) + tid] = localHistStart[tid];
    }
}

//------------------------------------------------------------
// kernel__radixPermute
//
// Purpose : Prefix sum results are used to scatter each work-group's elements to their correct position.
//------------------------------------------------------------

#if defined(OCL_DEVICE_GPU) && defined(OCL_PLATFORM_NVIDIA)

__kernel
void kernel__radixPermute(
	__global const KV_TYPE* dataIn,		// size 4*4 int2s per block
	__global KV_TYPE* dataOut,				// size 4*4 int2s per block
	__global const int* histSum,		// size 16 per block (64 B)
	__global const int* blockHists,		// size 16 int2s per block (64 B)
	const int bitOffset,				// k*4, k=0..7
	const int N,
	const int numBlocks)
{    
    const int tid = get_local_id(0);	
	const int groupId = get_group_id(0);
    const int4 tid4 = ((const int4)tid) + (const int4)(0,WGZ,WGZ_x2,WGZ_x3);		
	const int4 gid4 = tid4 + ((const int4)groupId<<2);
	
    __local int sharedHistSum[16];
    __local int localHistStart[16];

    // Fetch per-block KV_TYPE histogram and int histogram sums
    if (tid < 16)
    {
        sharedHistSum[tid] = histSum[tid * numBlocks + groupId];
        localHistStart[tid] = blockHists[(groupId << 5) + tid]; // groupId * 32 + tid
		//localHistStart[tid] = blockHists[(groupId << 4) + tid]; // groupId * 16 + tid
    }
	
	BARRIER_LOCAL;
	
	KV_TYPE myData;
    int myShiftedKeys;
	int finalOffset;	

	myData = (gid4.x < N) ? dataIn[gid4.x] : MAX_KV_TYPE;
	//myData = dataIn[gid4.x];
    myShiftedKeys = EXTRACT_KEY_4BITS(myData, bitOffset);
	finalOffset = tid4.x - localHistStart[myShiftedKeys] + sharedHistSum[myShiftedKeys];
	if (finalOffset < N) dataOut[finalOffset] = myData;
	
	myData = (gid4.y < N) ? dataIn[gid4.y] : MAX_KV_TYPE;
	//myData = dataIn[gid4.y];
    myShiftedKeys = EXTRACT_KEY_4BITS(myData, bitOffset);
	finalOffset = tid4.y - localHistStart[myShiftedKeys] + sharedHistSum[myShiftedKeys];
	if (finalOffset < N) dataOut[finalOffset] = myData;
	
	myData = (gid4.z < N) ? dataIn[gid4.z] : MAX_KV_TYPE;
	//myData = dataIn[gid4.z];
    myShiftedKeys = EXTRACT_KEY_4BITS(myData, bitOffset);
	finalOffset = tid4.z - localHistStart[myShiftedKeys] + sharedHistSum[myShiftedKeys];
	if (finalOffset < N) dataOut[finalOffset] = myData;

	myData = (gid4.w < N) ? dataIn[gid4.w] : MAX_KV_TYPE;	
	//myData = dataIn[gid4.w];
    myShiftedKeys = EXTRACT_KEY_4BITS(myData, bitOffset);
    finalOffset = tid4.w - localHistStart[myShiftedKeys] + sharedHistSum[myShiftedKeys];
    if (finalOffset < N) dataOut[finalOffset] = myData;
}

#else

__kernel
void kernel__radixPermute(
	__global const KV_TYPE* dataIn,		// size 4*4 int2s per block
	__global KV_TYPE* dataOut,				// size 4*4 int2s per block
	__global const int* histSum,		// size 16 per block (64 B)
	__global const int* blockHists,		// size 16 int2s per block (64 B)
	const uint bitOffset,				// k*4, k=0..7
	const uint N,
	const int numBlocks)
{
    const int4 gid4 = ((const int4)(get_global_id(0) << 2)) + (const int4)(0,1,2,3);
    const int tid = get_local_id(0);
    const int4 tid4 = ((int4)(tid << 2)) + (int4)(0,1,2,3);
    const int blockId = get_group_id(0);
    //const int numBlocks = get_num_groups(0); // Can be passed as a parameter !
    __local int sharedHistSum[16];
    __local int localHistStart[16];

    // Fetch per-block KV_TYPE histogram and int histogram sums
    if (tid < 16)
    {
        sharedHistSum[tid] = histSum[tid * numBlocks + blockId];
		//localHistStart[tid] = blockHists[(blockId << 4) + tid];
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

#endif