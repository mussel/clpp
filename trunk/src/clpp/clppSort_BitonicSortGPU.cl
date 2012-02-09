// Adapted from the Eric Bainville code.
//
// Copyright (c) Eric Bainville - June 2011
// http://www.bealto.com/gpu-sorting_intro.html

#if KEYS_ONLY
	#define getKey(a) (a)
	#define getValue(a) (0)
	#define makeData(k,v) (k)
#else
	#define getKey(a) ((a).x)
	#define getValue(a) ((a).y)
	#define makeData(k,v) ((uint2)((k),(v)))
#endif

#ifndef BLOCK_FACTOR
#define BLOCK_FACTOR 1
#endif

#define ORDER(a,b) { bool swap = reverse ^ (getKey(a)<getKey(b)); KV_TYPE auxa = a; KV_TYPE auxb = b; a = (swap)?auxb:auxa; b = (swap)?auxa:auxb; }

// N/2 threads
__kernel
void ParallelBitonic_B2(__global KV_TYPE* data, int inc, int dir, uint datasetSize)
{
	int t = get_global_id(0); // thread index
	int low = t & (inc - 1); // low order bits (below INC)
	int i = (t<<1) - low; // insert 0 at position INC
	bool reverse = ((dir & i) == 0); // asc/desc order
	data += i; // translate to first value

	// Load
	KV_TYPE x0 = data[  0];
	KV_TYPE x1 = data[inc];

	// Sort
	ORDER(x0,x1)

	// Store
	data[0  ] = x0;
	data[inc] = x1;
}

// N/4 threads
__kernel
void ParallelBitonic_B4(__global KV_TYPE * data,int inc,int dir, uint datasetSize)
{
	inc >>= 1;
	int t = get_global_id(0); // thread index
	int low = t & (inc - 1); // low order bits (below INC)
	int i = ((t - low) << 2) + low; // insert 00 at position INC
	bool reverse = ((dir & i) == 0); // asc/desc order
	data += i; // translate to first value
	
	// Load
	KV_TYPE x0 = data[    0];
	KV_TYPE x1 = data[  inc];
	KV_TYPE x2 = data[2*inc];
	KV_TYPE x3 = data[3*inc];
	
	// Sort
	ORDER(x0,x2)
	ORDER(x1,x3)
	ORDER(x0,x1)
	ORDER(x2,x3)
	
	// Store
	data[    0] = x0;
	data[  inc] = x1;
	data[2*inc] = x2;
	data[3*inc] = x3;
}

#define ORDERV(x,a,b) { bool swap = reverse ^ (getKey(x[a])<getKey(x[b])); KV_TYPE auxa = x[a]; KV_TYPE auxb = x[b]; x[a] = (swap)?auxb:auxa; x[b] = (swap)?auxa:auxb; }
#define B2V(x,a) { ORDERV(x,a,a+1) }
#define B4V(x,a) { for (int i4=0;i4<2;i4++) { ORDERV(x,a+i4,a+i4+2) } B2V(x,a) B2V(x,a+2) }
#define B8V(x,a) { for (int i8=0;i8<4;i8++) { ORDERV(x,a+i8,a+i8+4) } B4V(x,a) B4V(x,a+4) }
#define B16V(x,a) { for (int i16=0;i16<8;i16++) { ORDERV(x,a+i16,a+i16+8) } B8V(x,a) B8V(x,a+8) }

// N/8 threads
__kernel
void ParallelBitonic_B8(__global KV_TYPE * data,int inc,int dir, uint datasetSize)
{
	inc >>= 2;
	int t = get_global_id(0); // thread index
	int low = t & (inc - 1); // low order bits (below INC)
	int i = ((t - low) << 3) + low; // insert 000 at position INC
	bool reverse = ((dir & i) == 0); // asc/desc order
	data += i; // translate to first value
	
	// Load
	KV_TYPE x[8];
	for (int k=0;k<8;k++) x[k] = data[k*inc];
	
	// Sort
	B8V(x,0)
	
	// Store
	for (int k=0;k<8;k++) data[k*inc] = x[k];
}

// N/16 threads
__kernel
void ParallelBitonic_B16(__global KV_TYPE * data,int inc,int dir, uint datasetSize)
{
	inc >>= 3;
	int t = get_global_id(0); // thread index
	int low = t & (inc - 1); // low order bits (below INC)
	int i = ((t - low) << 4) + low; // insert 0000 at position INC
	bool reverse = ((dir & i) == 0); // asc/desc order
	data += i; // translate to first value
	
	// Load
	KV_TYPE x[16];
	for (int k=0;k<16;k++) x[k] = data[k*inc];
	
	// Sort
	B16V(x,0)
	
	// Store
	for (int k=0;k<16;k++) data[k*inc] = x[k];
}

__kernel
void ParallelBitonic_C4(__global KV_TYPE * data, int inc0, int dir, __local KV_TYPE* aux, uint datasetSize)
{
	int t = get_global_id(0); // thread index
	int wgBits = 4 * get_local_size(0) - 1; // bit mask to get index in local memory AUX (size is 4*WG)
	int inc,low,i;
	bool reverse;
	KV_TYPE x[4];
	
	// First iteration, global input, local output
	inc = inc0>>1;
	low = t & (inc - 1); // low order bits (below INC)
	i = ((t - low) << 2) + low; // insert 00 at position INC
	reverse = ((dir & i) == 0); // asc/desc order
	for (int k = 0; k < 4; k++) x[k] = data[i+k*inc];
	B4V(x,0);
	for (int k = 0; k < 4; k++) aux[(i+k*inc) & wgBits] = x[k];
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Internal iterations, local input and output
	for(;inc > 1; inc >>= 2)
	{
		low = t & (inc - 1); // low order bits (below INC)
		i = ((t - low) << 2) + low; // insert 00 at position INC
		reverse = ((dir & i) == 0); // asc/desc order
		for (int k=0;k<4;k++) x[k] = aux[(i+k*inc) & wgBits];
		B4V(x,0);
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int k=0;k<4;k++) aux[(i+k*inc) & wgBits] = x[k];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	// Final iteration, local input, global output, INC=1
	i = t << 2;
	reverse = ((dir & i) == 0); // asc/desc order
	for (int k = 0;k < 4; k++) x[k] = aux[(i+k) & wgBits];
	B4V(x,0);
	for (int k = 0;k < 4; k++) data[i+k] = x[k];
}