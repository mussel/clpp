//------------------------------------------------------------
// Purpose :
// ---------
//
// Algorithm :
// -----------
// Bitsonic sort, useful for small sets.
//
// References :
// ------------
// http://www.bealto.com/gpu-sorting_intro.html
// http://www-etud.iro.umontreal.ca/~blancher/comp599_mp/BitonicSort_b.cl
//------------------------------------------------------------

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
#define KEY(DATA) (DATA.s0)
#endif

#define BARRIER_LOCAL barrier(CLK_LOCAL_MEM_FENCE)

__kernel
void kernel__BitonicSort(__global KV_TYPE* taskIndices, const uint stage, const uint passOfStage)
{
	uint sortIncreasing = 1; // Direction
	uint gid = get_global_id(0);

	uint pairDistance = 1 << (stage - passOfStage);
	uint blockWidth = 2 * pairDistance;

	uint leftId = (gid % pairDistance) + (gid / pairDistance) * blockWidth;
	uint rightId = leftId + pairDistance;

	KV_TYPE leftElement = taskIndices[leftId];
	KV_TYPE rightElement = taskIndices[rightId];
	
	uint sameDirectionBlockWidth = 1 << stage;
	
	sortIncreasing = ((gid/sameDirectionBlockWidth) % 2 == 1) ? (1 - sortIncreasing) : sortIncreasing;

	uint leftKey = KEY(taskIndices[leftId]);
	uint rightKey = KEY(taskIndices[rightId]);
	
	KV_TYPE greater = leftKey > rightKey ? leftElement : rightElement;
    KV_TYPE lesser = leftKey > rightKey ? rightElement : leftElement;
	
	taskIndices[leftId] = sortIncreasing ? lesser : greater;    
    taskIndices[rightId] = sortIncreasing ? greater : lesser;
}