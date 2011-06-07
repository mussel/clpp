//------------------------------------------------------------
// Purpose :
// ---------
// Prefix sum or prefix scan is an operation where each output element contains the sum of all input elements preceding it.
//
// References :
// ------------
// http://graphics.idav.ucdavis.edu/publications/print_pub?pub_id=915
//
//------------------------------------------------------------

#pragma OPENCL EXTENSION cl_amd_printf : enable

#define T int
#define OPERATOR_APPLY(A,B) A+B
#define OPERATOR_IDENTITY 0

//#define BLOCK_SIZE 16
#define BLOCK_SIZE 32
//#define BLOCK_SIZE 64

//#ifdef OCL_PLATFORM==NVIDIA || OCL_GPU==1
//#define SYNC() 
//#endif

#define SYNC() barrier(CLK_LOCAL_MEM_FENCE)

//------------------------------------------------------------
// kernel__scanIntra
//
// Purpose : do a scan on a chunck of data.
//------------------------------------------------------------

__kernel
void kernel__scanIntra(__global T* input, __global T* sums, uint size)
{
	size_t idx = get_global_id(0);
	const uint lane = get_local_id(0);
	const uint bid = get_group_id(0);
	
	if (lane >= 1  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 1] , input[idx]);
	SYNC();
	if (lane >= 2  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 2] , input[idx]);
	SYNC();
	if (lane >= 4  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 4] , input[idx]);
	SYNC();
	if (lane >= 8  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 8] , input[idx]);
	SYNC();
	if (lane >= 16 && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 16], input[idx]);
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Store the sum		
	if (lane > 30)
		sums[bid] = input[idx];
}

//------------------------------------------------------------
// kernel__scanInter
//
// Purpose : do a scan on a chunck of data.
//------------------------------------------------------------

inline T scanIntra_exclusive(__global T* input, size_t idx, uint size)
{
	const uint lane = get_local_id(0);
	const uint bid = get_group_id(0);
	
	if (lane >= 1  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 1] , input[idx]);
	SYNC();
	if (lane >= 2  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 2] , input[idx]);
	SYNC();
	if (lane >= 4  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 4] , input[idx]);
	SYNC();
	if (lane >= 8  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 8] , input[idx]);
	SYNC();
	if (lane >= 16 && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 16], input[idx]);
	SYNC();
		
	return (lane > 0) ? input[idx-1] : OPERATOR_IDENTITY;
}

inline T scanIntra_inclusive(__global volatile T* input, size_t idx, uint size)
{
	const uint lane = get_local_id(0);
	const uint bid = get_group_id(0);
	
	if (lane >= 1  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 1] , input[idx]);
	SYNC();
	if (lane >= 2  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 2] , input[idx]);
	SYNC();
	if (lane >= 4  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 4] , input[idx]);
	SYNC();
	if (lane >= 8  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 8] , input[idx]);
	SYNC();
	if (lane >= 16 && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 16], input[idx]);
	SYNC();
		
	return input[idx];
}

T scan_block(__global T* ptr, uint size)
{
	size_t idx = get_global_id(0);
	const uint lane = get_local_id(0); // idx & 31;
	const uint bid = get_group_id(0); // idx >> 5;
	
	// Step 1: Intra-warp scan in each warp
	T val = scanIntra_exclusive(ptr, idx, size);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Step 2: Collect per-warp partial results
	if (lane == 31) ptr[bid] = ptr[idx];
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Step 3: Use 1st warp to scan per-warp results
	if (bid == 0) scanIntra_inclusive(ptr, idx, size);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Step 4: Accumulate results from Steps 1 and 3
	if (bid > 0) val = OPERATOR_APPLY(ptr[bid-1], val);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Step 5: Write and return the final result
	ptr[idx] = val;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	return val;
}

__kernel
void kernel__scan_block_anylength(
	__global T *ptr,
	__global const T *in,
	__global T *out,
	const uint B,
	const uint size
)
{
	size_t idx = get_global_id(0);
	const uint bidx = get_group_id(0);
	const uint TC = get_local_size(0);
	
	const uint nPasses = (float)ceil( ((float)B) / ((float)TC) );
	T reduceValue = OPERATOR_IDENTITY;
	
	for(uint i = 0; i < nPasses; ++i)
	{
		const uint offset = i * TC + (bidx * B);
		
		// Step 1: Read TC elements from global (off-chip)
		// memory to shared memory (on-chip)
		T input = ptr[idx] = in[offset + idx];
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// Step 2: Perform scan on TC elements
		T val = scan_block(ptr, size);
		
		// Step 3: Propagate reduced result from previous block
		// of TC elements
		val = OPERATOR_APPLY(val, reduceValue);
		
		// Step 4: Write out data to global memory
		out[offset + idx] = val;
		
		// Step 5: Choose reduced value for next iteration
		if (idx == (TC-1))
		{
			//ptr[idx] = (Kind == exclusive) ? OPERATOR_APPLY(input, val) : val;
			ptr[idx] = OPERATOR_APPLY(input, val);
		}		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		reduceValue = ptr[TC-1];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

//------------------------------------------------------------
// kernel__UniformAdd
//
// Purpose :
// Final step of large-array scan: combine basic inclusive scan with exclusive scan of top elements of input arrays.
//------------------------------------------------------------

__kernel
void kernel__UniformAdd(
	__global T* output,
	__global const T* blockSums,
	const uint outputSize
	)
{
    uint gid = get_global_id(0) * 2;
    const uint tid = get_local_id(0);
    const uint blockId = get_group_id(0);
	
	// Intel SDK fix
	//output[gid] += blockSums[blockId];
	//output[gid+1] += blockSums[blockId];

    __local T localBuffer[1];

    if (tid < 1)
        localBuffer[0] = (blockId < 1) ? OPERATOR_IDENTITY : blockSums[blockId - 1];

    barrier(CLK_LOCAL_MEM_FENCE);
	
	if (gid < outputSize)
		output[gid] += localBuffer[0];
	gid++;
	if (gid < outputSize)
		output[gid] += localBuffer[0];
}