#define T int

#define KEY_VALUE_1 1
#define KEY_VALUE_2 2
#define KEY_VALUE_3 4
#define KEY_VALUE_4 8
#define KEY_VALUE_5 16

__kernel
void kernel__Count(
	__global T* dataSet,
	__global int* countingBlocks,
	uint countings,
	uint valuesPerWorkgroup,
	uint datasetSize
)
{
	size_t gid = get_global_id(0);
	
	if (gid >= datasetSize)
		return;
		
	uint blockOffset = countings * get_group_id(0);
	uint valuesOffset = get_group_id(0);
	
	// Each workgroup will count 'valuesPerWorkgroup' elements
	for(uint i = 0; i < valuesPerWorkgroup; i++)
	{
		switch(dataSet[valuesOffset + i])
		{
			case KEY_VALUE_1:
				countingBlocks[blockOffset]++;
				break;
			case KEY_VALUE_2:
				countingBlocks[blockOffset+1]++;
				break;
			case KEY_VALUE_3:
				countingBlocks[blockOffset+2]++;
				break;
			case KEY_VALUE_4:
				countingBlocks[blockOffset+3]++;
				break;
			case KEY_VALUE_5:
				countingBlocks[blockOffset+4]++;
				break;
		}
	}
}