//------------------------------------------------------------
// Purpose :
// ---------
// Prefix sum or prefix scan is an operation where each output element contains the sum of all input elements preceding it.
//
// References :
// ------------
// Efficient Parallel Scan Algorithms for Many-core GPUs : http://graphics.idav.ucdavis.edu/publications/print_pub?pub_id=1041
//
//------------------------------------------------------------

#pragma OPENCL EXTENSION cl_amd_printf : enable

#define T int
#define OPERATOR_APPLY(A,B) A+B
#define OPERATOR_INVERSE_APPLY(A,B) A-B
#define OPERATOR_IDENTITY 0

//#define SIMT_SIZE 32
//#define SIMT_SIZE 64

//------------------------------------------------------------
// kernel__scan_exclusive
//
// Purpose : do an exclusive scan on a chunck of data.
//------------------------------------------------------------

__kernel
void kernel__scan_exclusive(__global volatile T* input, __global volatile T* sums, uint size)
{
	size_t idx = get_global_id(0);
	const uint lane = get_local_id(0);
	const uint bid = get_group_id(0);
	
	if (lane >= 1  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 1] , input[idx]);
	if (lane >= 2  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 2] , input[idx]);
	if (lane >= 4  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 4] , input[idx]);
	if (lane >= 8  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 8] , input[idx]);
	if (lane >= 16 && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 16], input[idx]);
	
#if SIMT_SIZE == 64
	if (lane >= 32  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 32] , input[idx]);
#endif
	
	// Store the sum		
#if SIMT_SIZE == 64
	if (lane == 63)
#else
	if (lane > 30)
#endif
		sums[bid] = input[idx];
}

//------------------------------------------------------------
// kernel__scan_inclusive
//
// Purpose : do an inclusive scan on a chunck of data.
//------------------------------------------------------------

__kernel
void kernel__scan_inclusive(__global volatile T* input, __global volatile T* sums, uint size)
{
	size_t idx = get_global_id(0);
	const uint lane = get_local_id(0);
	const uint bid = get_group_id(0);
	
	if (lane >= 1  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 1] , input[idx]);
	if (lane >= 2  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 2] , input[idx]);
	if (lane >= 4  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 4] , input[idx]);
	if (lane >= 8  && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 8] , input[idx]);
	if (lane >= 16 && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 16], input[idx]);
	
#if SIMT_SIZE == 64
	if (lane >= 32 && idx < size) input[idx] = OPERATOR_APPLY(input[idx - 32], input[idx]);
#endif

	// Store the sum		
#if SIMT_SIZE == 64
	if (lane == 63)
#else
	if (lane > 30)
#endif
		sums[bid] = input[idx];
}

//------------------------------------------------------------
// kernel__UniformAdd
//
// Purpose :
// Final step of large-array scan: combine basic inclusive scan with exclusive scan of top elements of input arrays.
//------------------------------------------------------------

__kernel
void kernel__UniformAdd_inclusive(
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

__kernel
void kernel__UniformAdd_exclusive(
	__global T* output,
	__global const T* blockSums,
	const uint outputSize,
	T zeroValue
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
        localBuffer[0] = (blockId < 1) ? OPERATOR_INVERSE_APPLY(OPERATOR_IDENTITY,zeroValue) : OPERATOR_INVERSE_APPLY(blockSums[blockId - 1],zeroValue);

    barrier(CLK_LOCAL_MEM_FENCE);
	
	if (gid < outputSize)
		output[gid] += localBuffer[0];
	gid++;
	if (gid < outputSize)
		output[gid] += localBuffer[0];
}