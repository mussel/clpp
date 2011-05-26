#include <stdlib.h>
#include <algorithm>

#include "clpp/clppSort_Blelloch.h"
#include "clpp/clppSort_CPU.h"
#include "clpp/clppSort_nvRadixSort.h"

using namespace std;

void makeRandomUint16Vector(unsigned short *a, unsigned int numElements, unsigned int keybits);
void makeRandomUint32Vector(unsigned int *a, unsigned int numElements, unsigned int keybits);
void benchmark(clppContext context, clppSort* sort, unsigned int* keys, unsigned int* keysSorted, unsigned int datasetSize);
bool checkIsSorted(unsigned int* sorted, unsigned int* tocheck, size_t datasetSize, string algorithmName);

int main(int argc, const char **argv)
{
	//unsigned int datasetSize = 8192;
	unsigned int datasetSize = 100000;

	//---- Create a new set of random datas
	unsigned int* keys = (unsigned int*)malloc(datasetSize * sizeof(int));
	makeRandomUint32Vector(keys, datasetSize, 32);

	unsigned int* keysCopy = (unsigned int*)malloc(datasetSize * sizeof(int));
	memcpy(keysCopy, keys, datasetSize * sizeof(int));

	// Create a copy
	unsigned int* keysSorted = (unsigned int*)malloc(datasetSize * sizeof(int));
	memcpy(keysSorted, keys, datasetSize * sizeof(int));

	// use standard sort
	sort(keysSorted, keysSorted + datasetSize);

	//---- Prepare a clpp Context
	clppContext context;
	context.setup(1, 0);

	//---- Start the benchmark
	clppSort* clppsort;

	// Brute fore
	clppsort = new clppSort_CPU(&context, "");
	benchmark(context, clppsort, keys, keysSorted, datasetSize);

	// Blelloch
	memcpy(keys, keysCopy, datasetSize * sizeof(int));
	clppsort = new clppSort_Blelloch(&context, "src/clpp/");
	benchmark(context, clppsort, keys, keysSorted, datasetSize);	

	// NV
	memcpy(keys, keysCopy, datasetSize * sizeof(int));
	clppsort = new clppSort_nvRadixSort(&context, "src/clpp/", datasetSize, 128); // 128 = work group size
	benchmark(context, clppsort, keys, keysSorted, datasetSize);

	//---- Free
	free(keys);
}

void benchmark(clppContext context, clppSort* sort, unsigned int* keys, unsigned int* keysSorted, unsigned int datasetSize)
{
	sort->pushDatas(keys, keys, 4, 4, datasetSize, 32);

	double start = sort->ClockTime();
	sort->sort();
	double delta = sort->ClockTime() - start;

	cout << "Performance for [" << sort->getName() << "] : data-set size[" << datasetSize << "] time (ms): " << delta << endl;

	sort->popDatas();

	checkIsSorted(keysSorted, keys, datasetSize, sort->getName());
}

void makeRandomUint16Vector(unsigned short *a, unsigned int numElements, unsigned int keybits)
{
    // Fill up with some random data
    int keyshiftmask = 0;
    if (keybits > 16) keyshiftmask = (1 << (keybits - 16)) - 1;
    int keymask = 0xffff;
    if (keybits < 16) keymask = (1 << keybits) - 1;

    srand(0);
    for(unsigned int i = 0; i < numElements; ++i)   
        a[i] = ((rand() & keyshiftmask)<<16); 
}

void makeRandomUint32Vector(unsigned int* a, unsigned int numElements, unsigned int keybits)
{
    // Fill up with some random data
    int keyshiftmask = 0;
    if (keybits > 16) keyshiftmask = (1 << (keybits - 16)) - 1;
    int keymask = 0xffff;
    if (keybits < 16) keymask = (1 << keybits) - 1;

    srand(95123);
    for(unsigned int i=0; i < numElements; ++i)   
        a[i] = ((rand() & keyshiftmask)<<16) | (rand() & keymask); 
}

bool checkIsSorted(unsigned int* sorted, unsigned int* tocheck, size_t datasetSize, string algorithmName)
{
	for(size_t i = 0; i < datasetSize; i++)
		if (sorted[i] != tocheck[i])
		{
			cout << "Algorithm FAILED : " << algorithmName << endl;
			return false;
		}

	return true;
}