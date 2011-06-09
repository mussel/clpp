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
// http://graphics.idav.ucdavis.edu/publications/print_pub?pub_id=1041
//------------------------------------------------------------

#pragma OPENCL EXTENSION cl_amd_printf : enable

#define T int
#define OPERATOR_APPLY(A,B) A+B
#define OPERATOR_IDENTITY 0

//------------------------------------------------------------
// kernel__scanInter
//
// Purpose : do a scan on a chunck of data.
//------------------------------------------------------------

inline T scan_simt_exclusive(__local T* input, size_t idx, uint size)
{
	const uint lane = idx & 31; // SIMT size
	
	if (lane >= 1  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 1] , input[idx]);
	if (lane >= 2  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 2] , input[idx]);
	if (lane >= 4  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 4] , input[idx]);
	if (lane >= 8  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 8] , input[idx]);
	if (lane >= 16 && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 16], input[idx]);
		
	return (lane > 0) ? input[idx-1] : OPERATOR_IDENTITY;
}

inline T scan_simt_inclusive(__local volatile T* input, size_t idx, uint size)
{
	const uint lane = idx & 31; // SIMT size
	
	if (lane >= 1  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 1] , input[idx]);
	if (lane >= 2  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 2] , input[idx]);
	if (lane >= 4  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 4] , input[idx]);
	if (lane >= 8  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 8] , input[idx]);
	if (lane >= 16 && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 16], input[idx]);
		
	return input[idx];
}

T scan_workgroup(__local T* localBuf, uint size)
{
	size_t idx = get_local_id(0);
	const uint lane = idx & 31;
	const uint simt_bid = idx >> 5;
	
	// Step 1: Intra-warp scan in each warp
	T val = scan_simt_exclusive(localBuf, idx, size);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Step 2: Collect per-warp partial results (the sum)
	if (lane == 31) localBuf[simt_bid] = localBuf[idx];
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Step 3: Use 1st warp to scan per-warp results
	if (simt_bid == 0) scan_simt_inclusive(localBuf, idx, size);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Step 4: Accumulate results from Steps 1 and 3
	if (simt_bid > 0) val = OPERATOR_APPLY(localBuf[simt_bid-1], val);
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Step 5: Write and return the final result
	localBuf[idx] = val;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	return val;
}

__kernel
void kernel__scan_block_anylength(
	__local T* localBuf,
	__global const T *in,
	__global T *out,
	const uint B,
	uint size,
	const uint nPasses
)
{	
	size_t idx = get_local_id(0);
	const uint bidx = get_group_id(0);
	const uint TC = get_local_size(0);
	
	T reduceValue = OPERATOR_IDENTITY;
	
	//__local T localBuf[384];
	
	for(uint i = 0; i < nPasses; ++i)
	{
		const uint offset = i * TC + (bidx * B);
		
		if ((offset + idx) > size-1) return;
		
		// Step 1: Read TC elements from global (off-chip) memory to local memory (on-chip)
		T input = localBuf[idx] = in[offset + idx];
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// Step 2: Perform scan on TC elements
		T val = scan_workgroup(localBuf, size);
		
		// Step 3: Propagate reduced result from previous block of TC elements
		val = OPERATOR_APPLY(val, reduceValue);
		
		// Step 4: Write out data to global memory
		out[offset + idx] = val;
		
		// Step 5: Choose reduced value for next iteration
		if (idx == (TC-1))
		{
			//localBuf[idx] = (Kind == exclusive) ? OPERATOR_APPLY(input, val) : val;
			localBuf[idx] = OPERATOR_APPLY(input, val);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		
		reduceValue = localBuf[TC-1];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}