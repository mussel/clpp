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
// http://graphics.idav.ucdavis.edu/publications/print_pub?pub_id=915
//
// Other references :
// ------------------
// http://developer.nvidia.com/node/57
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
	
	//barrier(CLK_LOCAL_MEM_FENCE);
	
	//idx++;
	
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