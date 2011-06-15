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

/*#ifdef cl_khr_byte_addressable_store
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define LTYPE ushort
#else
#define LTYPE uint
#endif*/
//#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
//#define LTYPE uint
/*#ifdef BYTE_ADDR
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define LTYPE int
#else
#define LTYPE int
#endif*/
#define LTYPE int	// TODO

//------------------------------------------------------------
// kernel__radixLocalSort
//
// Purpose : 
//------------------------------------------------------------

__kernel
void kernel__radixLocalSort(
	__local int2* shared,      // size 4*4 int2s (8 kB)
	__local LTYPE* indices,    // size 4*4 shorts (4 kB)
	__local LTYPE* sharedSum,  // size 4*4*2 shorts (2 kB)
	__global int2* data,       // size 4*4 int2s per block (8 kB)
	__global int* hist,        // size 16  per block (64 B)
	__global int* blockHists, // size 16 int2s per block (64 B)
	const int bitOffset,       // k*4, k=0..7
	const int N)              // N = 32 (32x int2 global)
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
    shared[tid4.x] = (gid4+0 < N) ? data[gid4+0] : (int2)0x7FFFFFFF;
    shared[tid4.y] = (gid4+1 < N) ? data[gid4+1] : (int2)0x7FFFFFFF;
    shared[tid4.z] = (gid4+2 < N) ? data[gid4+2] : (int2)0x7FFFFFFF;
    shared[tid4.w] = (gid4+3 < N) ? data[gid4+3] : (int2)0x7FFFFFFF;

    indices[tid4.x] = tid4.x;
    indices[tid4.y] = tid4.y;
    indices[tid4.z] = tid4.z;
    indices[tid4.w] = tid4.w;

    int srcBase = 0;
    int dstBase = blockSize4;
	
	//-------- 1) 4 x local 1-bit split
	// Should find a way to use createBestScan to improve performance here

    uint shift = bitOffset;
    for (int i = 0; i < 4; i++, shift++)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        sharedSum[tid4.x] = (LTYPE)1 - ((LTYPE)(((int)shared[indices[srcBase + tid4.x]].x) >> shift) & 0x1);
        sharedSum[tid4.y] = (LTYPE)1 - ((LTYPE)(((int)shared[indices[srcBase + tid4.y]].x) >> shift) & 0x1);
        sharedSum[tid4.z] = (LTYPE)1 - ((LTYPE)(((int)shared[indices[srcBase + tid4.z]].x) >> shift) & 0x1);
        sharedSum[tid4.w] = (LTYPE)1 - ((LTYPE)(((int)shared[indices[srcBase + tid4.w]].x) >> shift) & 0x1);

        // local serial reduction
        sharedSum[tid4.y] += sharedSum[tid4.x];
        sharedSum[tid4.w] += sharedSum[tid4.z];
        sharedSum[tid4.w] += sharedSum[tid4.y];

        int offset = 2;
        for (int d = blockSize >> 1; d > 0; d >>= 1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            if (tid < d)
            {
                int ai = (((tid << 1) + 1) << offset) - 1;
                int bi = (((tid << 1) + 2) << offset) - 1;
                sharedSum[bi] += sharedSum[ai];
            }
            offset++;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid == blockSize-1)
        {
            incSum[0] = sharedSum[tid4.w];
            sharedSum[tid4.w] = 0;
        }

        // expansion
        for (int d = 1; d < blockSize; d <<= 1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset--;
            if (tid < d)
            {
                int ai = (((tid << 1) + 1) << offset) - 1;
                int bi = (((tid << 1) + 2) << offset) - 1;
                LTYPE tmp = sharedSum[ai];
                sharedSum[ai] = sharedSum[bi];
                sharedSum[bi] += tmp;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
		
        // local 4-element serial expansion
        LTYPE tmp;
        tmp = sharedSum[tid4.y];    sharedSum[tid4.y] = sharedSum[tid4.w];  sharedSum[tid4.w] += tmp;
        tmp = sharedSum[tid4.x];    sharedSum[tid4.x] = sharedSum[tid4.y];  sharedSum[tid4.y] += tmp;
        tmp = sharedSum[tid4.z];    sharedSum[tid4.z] = sharedSum[tid4.w];  sharedSum[tid4.w] += tmp;

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
// Purpose : 
//------------------------------------------------------------

__kernel
void kernel__radixPermute(
	__global const int2* dataIn,     // size 4*4 int2s per block
	__global int2* dataOut,    // size 4*4 int2s per block
	__global const int* histSum,     // size 16  per block (64 B)
	__global const int* blockHists, // size 16 int2s per block (64 B)
	const int bitOffset,       // k*4, k=0..7
	const int N)              // N = 32 (32x int2 global)
{
    const int4 gid4 = ((const int4)(get_global_id(0) << 2)) + (const int4)(0,1,2,3);
    const int tid = get_local_id(0);
    const int4 tid4 = ((int4)(tid << 2)) + (int4)(0,1,2,3);
    const int blockId = get_group_id(0);
    const int blockSize = get_local_size(0);
    const int numBlocks = get_num_groups(0);
    __local int sharedHistSum[16];
    __local int localHistStart[16];

    // first, fetch per-block int2 histogram and int histogram sums
    if (tid < 16)
    {
        sharedHistSum[tid] = histSum[tid * numBlocks + blockId];
        localHistStart[tid] = blockHists[(blockId << 5) + tid];
    }

    // Copy data, each thread copies 4 (Cell,Tri) pairs into local shared mem
    barrier(CLK_LOCAL_MEM_FENCE);
    int2 myData[4];
    int myShiftedKeys[4];
    myData[0] = (gid4.x < N) ? dataIn[gid4.x] : (int2)0x7FFFFFFF;
    myData[1] = (gid4.y < N) ? dataIn[gid4.y] : (int2)0x7FFFFFFF;
    myData[2] = (gid4.z < N) ? dataIn[gid4.z] : (int2)0x7FFFFFFF;
    myData[3] = (gid4.w < N) ? dataIn[gid4.w] : (int2)0x7FFFFFFF;

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