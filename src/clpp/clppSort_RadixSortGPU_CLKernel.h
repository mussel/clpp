
char clCode_clppSort_RadixSortGPU[]=
"#pragma OPENCL EXTENSION cl_amd_printf : enable\n"
"#define WGZ 32\n"
"#define WGZ_x2 (WGZ*2)\n"
"#define WGZ_x3 (WGZ*3)\n"
"#define WGZ_x4 (WGZ*4)\n"
"#define WGZ_1 (WGZ-1)\n"
"#define WGZ_2 (WGZ-2)\n"
"#define WGZ_x2_1 (WGZ_x2-1)\n"
"#define WGZ_x3_1 (WGZ_x3-1)\n"
"#define WGZ_x4_1 (WGZ_x4-1)\n"
"#define WGZ_x4_2 (WGZ_x4-2)\n"
"#ifdef KEYS_ONLY\n"
"#define KEY(DATA) (DATA)\n"
"#else\n"
"#define KEY(DATA) (DATA.x)\n"
"#endif\n"
"#define EXTRACT_KEY_BIT(VALUE,BIT) ((KEY(VALUE)>>BIT)&0x1)\n"
"#define EXTRACT_KEY_4BITS(VALUE,BIT) ((KEY(VALUE)>>BIT)&0xF)\n"
"#define BARRIER_LOCAL barrier(CLK_LOCAL_MEM_FENCE)\n"
"#define SIMT 32\n"
"#define SIMT_1 (SIMT-1)\n"
"#define SIMT_2 (SIMT-2)\n"
"#define COMPUTE_UNITS 4\n"
"#define TPG (COMPUTE_UNITS * SIMT)\n"
"#define TPG_2 (TPG-2)\n"
"inline \n"
"uint4 inclusive_scan_128(volatile __local uint* localBuffer, const uint tid, uint block, uint lane, uint4 initialValue, __local uint* bitsOnCount)\n"
"{	\n"
"	//---- scan : 4 bits\n"
"	uint4 localBits = initialValue;\n"
"	localBits.y += localBits.x;\n"
"	localBits.z += localBits.y;\n"
"	localBits.w += localBits.z;\n"
"	\n"
"	//---- scan the last 4x32 bits (The sum in the previous scan)\n"
"	\n"
"	// The following is the same as 2 * SIMT_SIZE * simtId + threadInSIMT = \n"
"	//uint tid2 = 2 * tid - lane;\n"
"	\n"
"	uint tid2 = block * 2 * SIMT + lane;\n"
"	\n"
"	localBuffer[tid2] = 0;\n"
"	tid2 += SIMT;\n"
"	localBuffer[tid2] = localBits.w;\n"
"	\n"
"	localBuffer[tid2] += localBuffer[tid2 - 1];\n"
"	localBuffer[tid2] += localBuffer[tid2 - 2];\n"
"	localBuffer[tid2] += localBuffer[tid2 - 4];\n"
"	localBuffer[tid2] += localBuffer[tid2 - 8];\n"
"	localBuffer[tid2] += localBuffer[tid2 - 16];\n"
"	\n"
"	//---- Add the sum to create a scan of 128 bits\n"
"	return localBits + localBuffer[tid2 - 1];\n"
"}\n"
"inline \n"
"uint4 exclusive_scan_512(volatile __local uint* localBuffer, const uint tid, uint4 initialValue, __local uint* bitsOnCount)\n"
"{\n"
"	uint lane = tid & SIMT_1;\n"
"	uint block = tid >> 5;\n"
"	\n"
"	uint4 localBits = inclusive_scan_128(localBuffer, tid, block, lane, initialValue, bitsOnCount);\n"
"	\n"
"	barrier(CLK_LOCAL_MEM_FENCE);\n"
"	\n"
"	//---- Scan 512\n"
"	if (lane > SIMT_2)\n"
"	{\n"
"		localBuffer[block] = 0;\n"
"		localBuffer[4 + block] = localBits.w;\n"
"	}\n"
"		\n"
"	barrier(CLK_LOCAL_MEM_FENCE);\n"
"	\n"
"	// Use the SIMT capabilities\n"
"	if (tid < 4)\n"
"	{\n"
"		uint tid2 = tid + 4;		\n"
"		localBuffer[tid2] += localBuffer[tid2 - 1];\n"
"		localBuffer[tid2] += localBuffer[tid2 - 2];\n"
"	}\n"
"	\n"
"	barrier(CLK_LOCAL_MEM_FENCE);\n"
"	\n"
"	// Add the sum\n"
"	localBits += localBuffer[block + 4 - 1];\n"
"	\n"
"	// Total number of '1' in the array, retreived from the inclusive scan\n"
"	if (tid > TPG_2)\n"
"		bitsOnCount[0] = localBits.w;\n"
"		\n"
"	// To exclusive scan\n"
"	return localBits - initialValue;\n"
"}\n"
"__kernel\n"
"void kernel__radixLocalSort(\n"
"	//__local KV_TYPE* localDataOLD,\n"
"	__global KV_TYPE* data,\n"
"	const int bitOffset,\n"
"	const int N)\n"
"{\n"
"	const uint tid = (uint)get_local_id(0);\n"
"	const uint4 tid4 = (const uint4)(tid << 2) + (const uint4)(0,1,2,3);\n"
"	const uint4 gid4 = (const uint4)(get_global_id(0) << 2) + (const uint4)(0,1,2,3);\n"
"	\n"
"	// Local memory\n"
"	__local KV_TYPE localDataArray[TPG*4*2]; // Faster than using it as a parameter !!!\n"
"	__local KV_TYPE* localData = localDataArray;\n"
"	__local KV_TYPE* localTemp = localData + TPG * 4;\n"
"__local uint bitsOnCount[1];\n"
"	__local uint localBuffer[TPG*2];\n"
"localData[tid4.x] = (gid4.x < N) ? data[gid4.x] : MAX_KV_TYPE;\n"
"localData[tid4.y] = (gid4.y < N) ? data[gid4.y] : MAX_KV_TYPE;\n"
"localData[tid4.z] = (gid4.z < N) ? data[gid4.z] : MAX_KV_TYPE;\n"
"localData[tid4.w] = (gid4.w < N) ? data[gid4.w] : MAX_KV_TYPE;\n"
"	\n"
"	//-------- 1) 4 x local 1-bit split	\n"
"	#pragma unroll\n"
"for(uint shift = bitOffset; shift < (bitOffset+4); shift++) // Radix 4\n"
"{\n"
"		//barrier(CLK_LOCAL_MEM_FENCE);\n"
"		\n"
"		//---- Setup the array of 4 bits (of level shift)\n"
"		// Create the '1s' array as explained at : http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html\n"
"		// In fact we simply inverse the bits	\n"
"		// Local copy and bits extraction\n"
"		uint4 flags;\n"
"		flags.x = ! EXTRACT_KEY_BIT(localData[tid4.x], shift);\n"
"flags.y = ! EXTRACT_KEY_BIT(localData[tid4.y], shift);\n"
"flags.z = ! EXTRACT_KEY_BIT(localData[tid4.z], shift);\n"
"flags.w = ! EXTRACT_KEY_BIT(localData[tid4.w], shift);\n"
"		//---- Do a scan of the 128 bits and retreive the total number of '1' in 'bitsOnCount'\n"
"		uint4 localBitsScan = exclusive_scan_512(localBuffer, tid, flags, bitsOnCount);\n"
"		\n"
"		// Waiting for 'bitsOnCount'\n"
"		barrier(CLK_LOCAL_MEM_FENCE);\n"
"		\n"
"		//---- Relocate to the right position	\n"
"		uint4 offset = (1-flags) * ((uint4)(bitsOnCount[0]) + tid4 - localBitsScan) + flags * localBitsScan;\n"
"		localTemp[offset.x] = localData[tid4.x];\n"
"		localTemp[offset.y] = localData[tid4.y];\n"
"		localTemp[offset.z] = localData[tid4.z];\n"
"		localTemp[offset.w] = localData[tid4.w];\n"
"		\n"
"		// Wait before swapping the 'local' buffer pointers. They are shared by the whole local context\n"
"		barrier(CLK_LOCAL_MEM_FENCE);\n"
"		// Swap the buffer pointers\n"
"		__local KV_TYPE* swBuf = localData;\n"
"		localData = localTemp;\n"
"		localTemp = swBuf;\n"
"}\n"
"	\n"
"	//barrier(CLK_LOCAL_MEM_FENCE);\n"
"	\n"
"	// Write sorted data back to global memory\n"
"	if (gid4.x < N) data[gid4.x] = localData[tid4.x];\n"
"if (gid4.y < N) data[gid4.y] = localData[tid4.y];\n"
"if (gid4.z < N) data[gid4.z] = localData[tid4.z];\n"
"if (gid4.w < N) data[gid4.w] = localData[tid4.w];\n"
"}\n"
"__kernel\n"
"void kernel__localHistogram(__global KV_TYPE* data, const int bitOffset, __global int* radixCount, __global int* radixOffsets, const int N)\n"
"{\n"
"const int tid = (int)get_local_id(0);\n"
"const int4 tid4 = (int4)(tid << 2) + (const int4)(0,1,2,3);\n"
"	const int4 gid4 = (int4)(get_global_id(0) << 2) + (const int4)(0,1,2,3);\n"
"	const int blockId = (int)get_group_id(0);\n"
"	\n"
"	__local uint localData[WGZ_x4];\n"
"	\n"
"	// Contains the 2 histograms (16 values)\n"
"__local int localHistStart[16]; // 2^4 = 16\n"
"__local int localHistEnd[16];\n"
"	\n"
"	//---- Extract the radix\n"
"localData[tid4.x] = (gid4.x < N) ? EXTRACT_KEY_4BITS(data[gid4.x], bitOffset) : 0xF; //EXTRACT_KEY_4BITS(MAX_KV_TYPE, bitOffset);\n"
"localData[tid4.y] = (gid4.y < N) ? EXTRACT_KEY_4BITS(data[gid4.y], bitOffset) : 0xF; //EXTRACT_KEY_4BITS(MAX_KV_TYPE, bitOffset);\n"
"localData[tid4.z] = (gid4.z < N) ? EXTRACT_KEY_4BITS(data[gid4.z], bitOffset) : 0xF; //EXTRACT_KEY_4BITS(MAX_KV_TYPE, bitOffset);\n"
"localData[tid4.w] = (gid4.w < N) ? EXTRACT_KEY_4BITS(data[gid4.w], bitOffset) : 0xF; //EXTRACT_KEY_4BITS(MAX_KV_TYPE, bitOffset);\n"
"	\n"
"	//---- Create the histogram\n"
"barrier(CLK_LOCAL_MEM_FENCE);\n"
"	\n"
"	// Reset the local histogram\n"
"if (tid < 16)\n"
"{\n"
"localHistStart[tid] = 0;\n"
"localHistEnd[tid] = -1;\n"
"}\n"
"	barrier(CLK_LOCAL_MEM_FENCE);\n"
"	\n"
"	// This way, for the first 'instance' of a radix, we store its index.\n"
"	// We also store where each radix ends in 'localHistEnd'.\n"
"	//\n"
"	// And so, if we use end-start+1 we have the histogram value to store.\n"
"	\n"
"if (tid4.x > 0 && localData[tid4.x] != localData[tid4.x-1])\n"
"{\n"
"		localHistStart[localData[tid4.x]] = tid4.x;\n"
"localHistEnd[localData[tid4.x-1]] = tid4.x - 1;        \n"
"}\n"
"	//BARRIER_LOCAL;\n"
"if (localData[tid4.y] != localData[tid4.x])\n"
"{\n"
"		localHistStart[localData[tid4.y]] = tid4.y;\n"
"localHistEnd[localData[tid4.x]] = tid4.x;        \n"
"}\n"
"	//BARRIER_LOCAL;\n"
"if (localData[tid4.z] != localData[tid4.y])\n"
"{\n"
"		localHistStart[localData[tid4.z]] = tid4.z;\n"
"localHistEnd[localData[tid4.y]] = tid4.y;        \n"
"}\n"
"	//BARRIER_LOCAL;\n"
"if (localData[tid4.w] != localData[tid4.z])\n"
"{\n"
"		localHistStart[localData[tid4.w]] = tid4.w;\n"
"localHistEnd[localData[tid4.z]] = tid4.z;\n"
"}\n"
"	//BARRIER_LOCAL;\n"
"	// First and last histogram values\n"
"if (tid < 1)\n"
"{\n"
"		localHistStart[localData[0]] = 0;\n"
"		localHistEnd[localData[WGZ_x4-1]] = WGZ_x4 - 1;		\n"
"}\n"
"barrier(CLK_LOCAL_MEM_FENCE);\n"
"	// Write the 16 histogram values to the global buffers\n"
"if (tid < 16)\n"
"{\n"
"radixCount[tid * get_num_groups(0) + blockId] = localHistEnd[tid] - localHistStart[tid] + 1;\n"
"		radixOffsets[(blockId << 4) + tid] = localHistStart[tid];\n"
"}\n"
"}\n"
"__kernel\n"
"void kernel__radixPermute(\n"
"	__global const KV_TYPE* dataIn,		// size 4*4 int2s per block\n"
"	__global KV_TYPE* dataOut,			// size 4*4 int2s per block\n"
"	__global const int* histSum,		// size 16 per block (64 B)\n"
"	__global const int* blockHists,		// size 16 int2s per block (64 B)\n"
"	const int bitOffset,				// k*4, k=0..7\n"
"	const int N,\n"
"	const int numBlocks)\n"
"{    \n"
"const int tid = get_local_id(0);	\n"
"	const int groupId = get_group_id(0);\n"
"const int4 tid4 = (int4)(tid << 2) + (const int4)(0,1,2,3);\n"
"	const int4 gid4 = (int4)(get_global_id(0) << 2) + (const int4)(0,1,2,3);\n"
"	\n"
"	\n"
"__local int sharedHistSum[16];\n"
"__local int localHistStart[16];\n"
"if (tid < 16)\n"
"{\n"
"sharedHistSum[tid] = histSum[tid * numBlocks + groupId];\n"
"localHistStart[tid] = blockHists[(groupId << 4) + tid]; // groupId * 32 + tid\n"
"}\n"
"	\n"
"	BARRIER_LOCAL;\n"
"	\n"
"	KV_TYPE myData;\n"
"int myShiftedKeys;\n"
"	int finalOffset;\n"
"	\n"
"	myData = (gid4.x < N) ? dataIn[gid4.x] : MAX_KV_TYPE;\n"
"myShiftedKeys = EXTRACT_KEY_4BITS(myData, bitOffset);\n"
"	finalOffset = tid4.x - localHistStart[myShiftedKeys] + sharedHistSum[myShiftedKeys];\n"
"	if (finalOffset < N) dataOut[finalOffset] = myData;\n"
"	\n"
"	myData = (gid4.y < N) ? dataIn[gid4.y] : MAX_KV_TYPE;\n"
"myShiftedKeys = EXTRACT_KEY_4BITS(myData, bitOffset);\n"
"	finalOffset = tid4.y - localHistStart[myShiftedKeys] + sharedHistSum[myShiftedKeys];\n"
"	if (finalOffset < N) dataOut[finalOffset] = myData;\n"
"	\n"
"	myData = (gid4.z < N) ? dataIn[gid4.z] : MAX_KV_TYPE;\n"
"myShiftedKeys = EXTRACT_KEY_4BITS(myData, bitOffset);\n"
"	finalOffset = tid4.z - localHistStart[myShiftedKeys] + sharedHistSum[myShiftedKeys];\n"
"	if (finalOffset < N) dataOut[finalOffset] = myData;\n"
"	myData = (gid4.w < N) ? dataIn[gid4.w] : MAX_KV_TYPE;	\n"
"myShiftedKeys = EXTRACT_KEY_4BITS(myData, bitOffset);\n"
"finalOffset = tid4.w - localHistStart[myShiftedKeys] + sharedHistSum[myShiftedKeys];\n"
"if (finalOffset < N) dataOut[finalOffset] = myData;\n"
"}\n"
;
