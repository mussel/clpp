#include <stdlib.h>
#include <algorithm>

#include "clpp/clppScan.h"
#include "clpp/clppSort_Blelloch.h"
#include "clpp/clppSort_CPU.h"
#include "clpp/clppSort_nvRadixSort.h"
#include "clpp/clppSort_Merill.h"

using namespace std;

void makeOneVector(unsigned int* a, unsigned int numElements);
void makeRandomUint16Vector(unsigned short *a, unsigned int numElements, unsigned int keybits);
void makeRandomUint32Vector(unsigned int *a, unsigned int numElements, unsigned int keybits);
void makeRandomUint32Vector_i(unsigned int *a, unsigned int numElements, unsigned int keybits);
void benchmark(clppContext context, clppSort* sort, unsigned int* keys, unsigned int* keysSorted, unsigned int datasetSize);
bool checkIsSorted(unsigned int* sorted, unsigned int* tocheck, size_t datasetSize, string algorithmName);
void benchmark_Scan(clppContext* context);
void benchmark_Sort(clppContext* context);

//unsigned int datasetSize = 1280000;
//unsigned int datasetSize = 128000;
//unsigned int datasetSize = 8192;
//unsigned int datasetSize = 131072;
//unsigned int datasetSize = 1<<10;
//unsigned int datasetSize = 1<<17;
//unsigned int datasetSize = 1<<19;
unsigned int datasetSize = 1<<21;
//unsigned int datasetSize = 1<<23;  // has to match _N for Blelloch ?

int main(int argc, const char** argv)
{
	clppProgram::setBasePath("src/clpp/");

	//---- Prepare a clpp Context
	clppContext context;
	context.setup(0, 0);

	benchmark_Scan(&context);
	//benchmark_Sort(&context);
}

void benchmark_Scan(clppContext* context)
{
	//---- Create a set of data
	unsigned int* values = (unsigned int*)malloc(datasetSize * sizeof(int));
	unsigned int* valuesOut = (unsigned int*)malloc(datasetSize * sizeof(int));
	makeOneVector(values, datasetSize);

	//--- Scan
	clppScan* scan = new clppScan(context, datasetSize);
	scan->pushDatas(values, valuesOut, sizeof(int), datasetSize);

	double start = scan->ClockTime();
	scan->scan();
	scan->waitCompletion();
	double delta = scan->ClockTime() - start;

	scan->popDatas();

	//---- Check the scan
	for(int i = 0; i < datasetSize; i++)
		if (valuesOut[i] != i)
		{
			cout << "Algorithm FAILED : Scan" << endl;
			break;
		}

	cout << "Performance for [Scan] : data-set size[" << datasetSize << "] time (ms): " << delta << endl;

	//---- Free
	free(values);
}

void benchmark_Sort(clppContext* context)
{
	//---- Create a new set of random datas
	unsigned int* keys = (unsigned int*)malloc(datasetSize * sizeof(int));
	makeRandomUint32Vector_i(keys, datasetSize, 32);

	unsigned int* keysCopy = (unsigned int*)malloc(datasetSize * sizeof(int));
	memcpy(keysCopy, keys, datasetSize * sizeof(int));

	// Create a copy
	unsigned int* keysSorted = (unsigned int*)malloc(datasetSize * sizeof(int));
	memcpy(keysSorted, keys, datasetSize * sizeof(int));

	// use standard sort
	sort(keysSorted, keysSorted + datasetSize);

	//---- Start the benchmark
	clppSort* clppsort;

	// Brute fore
	clppsort = new clppSort_CPU(context);
	benchmark(*context, clppsort, keys, keysSorted, datasetSize);

	// Blelloch
	memcpy(keys, keysCopy, datasetSize * sizeof(int));
	clppsort = new clppSort_Blelloch(context, datasetSize);
	benchmark(*context, clppsort, keys, keysSorted, datasetSize);	

	// NV
	memcpy(keys, keysCopy, datasetSize * sizeof(int));
	clppsort = new clppSort_nvRadixSort(context, datasetSize, 128); // 128 = work group size
	benchmark(*context, clppsort, keys, keysSorted, datasetSize);

	// Merill
	//memcpy(keys, keysCopy, datasetSize * sizeof(int));
	//clppsort = new clppSort_Merill(context, datasetSize); // 128 = work group size
	//benchmark(*context, clppsort, keys, keysSorted, datasetSize);

	//---- Free
	free(keys);
}

void benchmark(clppContext context, clppSort* sort, unsigned int* keys, unsigned int* keysSorted, unsigned int datasetSize)
{
 	sort->pushDatas(keys, keys, 4, 4, datasetSize, 32);

	double start = sort->ClockTime();
	sort->sort();
	sort->waitCompletion();
	double delta = sort->ClockTime() - start;

	cout << "Performance for [" << sort->getName() << "] : data-set size[" << datasetSize << "] time (ms): " << delta << endl;

	sort->popDatas();

	checkIsSorted(keysSorted, keys, datasetSize, sort->getName());
}

void makeOneVector(unsigned int* a, unsigned int numElements)
{
    for(unsigned int i = 0; i < numElements; ++i)   
        a[i] = 1; 
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

void makeRandomUint32Vector_i(unsigned int* a, unsigned int numElements, unsigned int keybits)
{
    for(unsigned int i=0; i < numElements; ++i)   
        a[i] = numElements - i; 
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
