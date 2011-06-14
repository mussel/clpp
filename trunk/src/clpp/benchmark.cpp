#include <stdlib.h>
#include <algorithm>

#include "clpp/StopWatch.h"
#include "clpp/clppScan.h"
#include "clpp/clppScan_Default.h"
#include "clpp/clppScan_GPU.h"
#include "clpp/clppScan_Merrill.h"

#include "clpp/clppSort_Blelloch.h"
#include "clpp/clppSort_CPU.h"
#include "clpp/clppSort_nvRadixSort.h"
#include "clpp/clppSort_RadixSort.h"
//#include "clpp/clppSort_Merill.h"

#include <string.h>

using namespace std;

void makeOneVector(unsigned int* a, unsigned int numElements);
void makeRandomUint16Vector(unsigned short *a, unsigned int numElements, unsigned int keybits);
void makeRandomUint32Vector(unsigned int *a, unsigned int numElements, unsigned int keybits);
void makeRandomUint32Vector_KV(unsigned int *a, unsigned int numElements, unsigned int keybits);
void makeRandomUint32Vector_i(unsigned int *a, unsigned int numElements, unsigned int keybits);

void benchmark_scan(clppContext* context, clppScan* scan, int datasetSize);
void benchmark_sort(clppContext context, clppSort* sort, unsigned int datasetSize);
void benchmark_sort_KV(clppContext context, clppSort* sort, unsigned int datasetSize);

bool checkIsSorted(unsigned int* tocheck, size_t datasetSize, string algorithmName);
bool checkIsSortedKV(unsigned int* tocheck, size_t datasetSize, string algorithmName);

void test_Scan(clppContext* context);
void test_Sort(clppContext* context);
void test_Sort_KV(clppContext* context);

//unsigned int datasetSize = 1280000;
//unsigned int datasetSize = 128000;
//unsigned int datasetSize = 8192;
//unsigned int datasetSize = 131072;
//unsigned int datasetSize = 1<<10;
//unsigned int datasetSize = 1<<17;
//unsigned int datasetSize = 1<<21;
//unsigned int datasetSize = _N;
//unsigned int datasetSize = 1<<23;  // has to match _N for Blelloch ?
//unsigned int datasetSize = 384000;
//unsigned int datasetSize = 400000;

unsigned int datasetSizes[7] = {128000, 256000, 512000, 1024000, 2048000, 4096000, 8196000};
unsigned int datasetSizesCount = 6;

StopWatch* stopWatcher = new StopWatch();

int main(int argc, const char** argv)
{
	clppProgram::setBasePath("src/clpp/");

	//---- Prepare a clpp Context
	clppContext context;
	context.setup(0, 0);

	// Scan
	//test_Scan(&context);

	// Sorting : key
	//test_Sort(&context);

	// Sorting : key + value
	test_Sort_KV(&context);
}

#pragma region test_Scan

void test_Scan(clppContext* context)
{
	//---- Default scan
	for(unsigned int i = 0; i < datasetSizesCount; i++)
	{
		clppScan* scan = new clppScan_Default(context, sizeof(int), datasetSizes[i]);
		benchmark_scan(context, scan, datasetSizes[i]);
		delete scan;
	}

	//---- GPU scan
	for(unsigned int i = 0; i < datasetSizesCount; i++)
	{
		clppScan* scan = new clppScan_GPU(context, sizeof(int), datasetSizes[i]);
		benchmark_scan(context, scan, datasetSizes[i]);
		delete scan;
	}

	//scan = new clppScan_Merrill(context, datasetSize);
	//benchmark_scan(context, scan);
}

#pragma endregion

#pragma region test_Sort

void test_Sort(clppContext* context)
{
	//---- Brute fore
	for(unsigned int i = 0; i < datasetSizesCount; i++)
	{
		clppSort* clppsort = new clppSort_CPU(context);
		benchmark_sort(*context, clppsort, datasetSizes[i]);
		delete clppsort;
	}

	//---- Blelloch
	for(unsigned int i = 0; i < datasetSizesCount; i++)
	{
		clppSort* clppsort = new clppSort_Blelloch(context, datasetSizes[i]);
		benchmark_sort(*context, clppsort, datasetSizes[i]);	
	}

	//---- NV
	for(unsigned int i = 0; i < datasetSizesCount; i++)
	{
		clppSort* clppsort = new clppSort_nvRadixSort(context, datasetSizes[i], 128); // 128 = work group size
		benchmark_sort(*context, clppsort, datasetSizes[i]);
	}

	// Merill
	//memcpy(keys, keysCopy, datasetSize * sizeof(int));
	//clppsort = new clppSort_Merill(context, datasetSize); // 128 = work group size
	//benchmark_sort(*context, clppsort, keys, keysSorted, datasetSize);
}

#pragma endregion

#pragma region test_Sort_KV

void test_Sort_KV(clppContext* context)
{
	//---- Radix sort
	for(unsigned int i = 0; i < datasetSizesCount; i++)
	{
		clppSort* clppsort = new clppSort_RadixSort(context, datasetSizes[i]);
		benchmark_sort_KV(*context, clppsort, datasetSizes[i]);
		delete clppsort;
	}
}

#pragma endregion

#pragma region benchmark_scan

void benchmark_scan(clppContext* context, clppScan* scan, int datasetSize)
{
	//---- Create a set of data
	unsigned int* values = (unsigned int*)malloc(datasetSize * sizeof(int));
	//makeOneVector(values, datasetSize);
	makeRandomUint32Vector(values, datasetSize, 32);

	//---- Scan : default
	unsigned int* cpuScanValues = (unsigned int*)malloc(datasetSize * sizeof(int));
	memcpy(cpuScanValues, values, datasetSize * sizeof(int));
	cpuScanValues[0] = 0;
	for(unsigned int i = 1; i < datasetSize; i++)
		cpuScanValues[i] = cpuScanValues[i-1] + values[i - 1];

	//---- Trace
	cout << "--------------- Scan : " << scan->getName() << endl;

	//--- Scan
	scan->pushDatas(values, datasetSize);

	stopWatcher->StartTimer();
	scan->scan();
	scan->waitCompletion();
	stopWatcher->StopTimer();

	scan->popDatas();

	//---- Check the scan
	for(int i = 0; i < datasetSize; i++)
		if (values[i] != cpuScanValues[i])
		{
			cout << "Algorithm FAILED : Scan" << endl;
			break;
		}

	cout << "Performance for [Scan] : data-set size[" << datasetSize << "] time (ms): " << stopWatcher->GetElapsedTime() << endl << endl;

	//---- Free
	free(values);
	free(cpuScanValues);
}

#pragma endregion

#pragma region benchmark_sort

void benchmark_sort(clppContext context, clppSort* sort, unsigned int datasetSize)
{
	//---- Create a new set of random datas
	unsigned int* keys = (unsigned int*)malloc(datasetSize * sizeof(int));
	//makeRandomUint32Vector_i(keys, datasetSize, 16);
	makeRandomUint32Vector(keys, datasetSize, 32);  

	//---- Trace
	cout << "--------------- Sort : " << sort->getName() << endl;

	//---- Push the datas
 	sort->pushDatas(keys, keys, 4, 4, datasetSize, 32);

	//---- Sort
	stopWatcher->StartTimer();
	sort->sort();
	sort->waitCompletion();
	stopWatcher->StopTimer();

	cout << "Performance for [" << sort->getName() << "] : data-set size[" << datasetSize << "] time (ms): " << stopWatcher->GetElapsedTime() << endl << endl;

	//---- Check if it is sorted
	sort->popDatas();
	checkIsSorted(keys, datasetSize, sort->getName());

	//---- Free
	free(keys);
}

#pragma endregion

#pragma region benchmark_sort_KV

void benchmark_sort_KV(clppContext context, clppSort* sort, unsigned int datasetSize)
{
	unsigned int* datas = (unsigned int*)malloc(2 * datasetSize * sizeof(int));
	makeRandomUint32Vector_KV(datas, datasetSize, 32);   

	//---- Trace
	cout << "--------------- Sort Key-Value : " << sort->getName() << endl;

	//---- Push the datas
 	sort->pushDatas(datas, datas, 4, 4, datasetSize, 32);

	//---- Sort
	stopWatcher->StartTimer();
	sort->sort();
	sort->waitCompletion();
	stopWatcher->StopTimer();

	cout << "Performance for [" << sort->getName() << "] : data-set size[" << datasetSize << "] time (ms): " << stopWatcher->GetElapsedTime() << endl << endl;

	//---- Check if it is sorted
	sort->popDatas();
	checkIsSortedKV(datas, datasetSize, sort->getName());

	//---- Free
	free(datas);
}

#pragma endregion

#pragma region make...

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
    // int keyshiftmask = 0;
    // if (keybits > 16) keyshiftmask = (1 << (keybits - 16)) - 1;
    // int keymask = 0xffff;
    // if (keybits < 16) keymask = (1 << keybits) - 1;

    srand(95123);
    //cout << "Warning, max int = "<< (1<<_TOTALBITS)<<endl;
    for(unsigned int i=0; i < numElements; ++i)  { 
      // a[i] = ((rand() & keyshiftmask)<<16) | (rand() & keymask); 
      a[i] = (rand()%(1<<_TOTALBITS));   // peut-être qu'il fallait pas :-)
    }
}

void makeRandomUint32Vector_KV(unsigned int* a, unsigned int numElements, unsigned int keybits)
{
    srand(95123);
    for(unsigned int i=0; i < numElements; ++i) 
	{
		a[i * 2 + 0] = (rand()%(1<<_TOTALBITS));
		a[i * 2 + 1] = i; //(rand()%(1<<_TOTALBITS));
    }
}

void makeRandomUint32Vector_i(unsigned int* a, unsigned int numElements, unsigned int keybits)
{
    for(unsigned int i=0; i < numElements; ++i)   
        a[i] = numElements - i; 
}

#pragma endregion

#pragma region checkIsSorted

bool checkIsSorted(unsigned int* tocheck, size_t datasetSize, string algorithmName)
{
	int previous = 0;
	for(size_t i = 0; i < datasetSize; i++)
	{
		if (previous > tocheck[i])
		{
			cout << "Algorithm FAILED : " << algorithmName << endl;
			return false;
		}
		previous = tocheck[i];
	}

	return true;
}

bool checkIsSortedKV(unsigned int* tocheck, size_t datasetSize, string algorithmName)
{
	int previous = 0;
	for(size_t i = 0; i < datasetSize; i++)
	{
		if (previous > tocheck[i*2])
		{
			cout << "Algorithm FAILED : " << algorithmName << endl;
			return false;
		}
		previous = tocheck[i*2];
	}

	return true;
}

#pragma endregion
