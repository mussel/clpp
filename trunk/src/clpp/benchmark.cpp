// In order to test that no value has been loosed ! Can take time to check !
#define PARAM_CHECK_HASLOOSEDVALUES 0
#define PARAM_BENCHMARK_LOOPS 20

// The number of bits to sort
#define PARAM_SORT_BITS 32

#include <stdlib.h>
#include <algorithm>

#include "clpp/StopWatch.h"
#include "clpp/clppScan.h"
#include "clpp/clppScan_Default.h"
#include "clpp/clppScan_GPU.h"

#include "clpp/clppSort_CPU.h"
#include "clpp/clppSort_RadixSort.h"
#include "clpp/clppSort_RadixSortGPU.h"

#include "clpp/clppCount.h"

#include <string.h>

using namespace std;

void makeOneVector(unsigned int* a, unsigned int numElements);
void makeRandomInt32Vector(unsigned int *a, unsigned int numElements, unsigned int keybits, bool keysOnly);

void benchmark_scan(clppContext* context, clppScan* scan, int datasetSize);
void benchmark_sort(clppContext context, clppSort* sort, unsigned int datasetSize, unsigned int bits);
void benchmark_sort_KV(clppContext context, clppSort* sort, unsigned int datasetSize, unsigned int bits);

bool checkIsSorted(unsigned int* tocheck, size_t datasetSize, string algorithmName, bool keysOnly, int sortId);
bool checkHasLooseDatasKV(int* unsorted, int* sorted, size_t datasetSize, string algorithmName);

void test_Scan(clppContext* context);
void test_Sort(clppContext* context);
void test_Sort_KV(clppContext* context);
void test_Count(clppContext* context);

//unsigned int datasetSizes[8] = {262144, 128000, 256000, 512000, 1024000, 2048000, 4096000, 8196000};
//unsigned int datasetSizes[8] = {16000, 128000, 256000, 512000, 1024000, 2048000, 4096000, 8196000};

// Small problems
unsigned int datasetSizes[10] = {100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000};

// Big problems
//unsigned int datasetSizes[10] = {16000000, 32000000, 48000000, 64000000, 80000000, 96000000, 112000000, 128000000, 144000000, 160000000};

unsigned int datasetSizesCount = 10;

StopWatch* stopWatcher = new StopWatch();

int main(int argc, const char** argv)
{
	clppProgram::setBasePath("src/clpp/");

	//---- Prepare a clpp Context
	clppContext context;
	context.setup(2, 0);
	context.printInformation();

	// Scan
	//test_Scan(&context);

	// Sorting : key
	test_Sort(&context);

	// Sorting : key + value
	test_Sort_KV(&context);

	// Count
	//test_Count(&context);
}

#pragma region test_Scan

void test_Scan(clppContext* context)
{
	//---- Default scan
	if (context->isCPU)
	{
		cout << "--------------- Scan : CPU scan" << endl;
		for(unsigned int i = 0; i < datasetSizesCount; i++)
		{
			clppScan* scan = new clppScan_Default(context, sizeof(int), datasetSizes[i]);
			benchmark_scan(context, scan, datasetSizes[i]);
			delete scan;
		}
	}

	//---- GPU scan
	if (context->isGPU)
	{
		cout << "--------------- Scan : GPU scan" << endl;
		for(unsigned int i = 0; i < datasetSizesCount; i++)
		{
			clppScan* scan = new clppScan_GPU(context, sizeof(int), datasetSizes[i]);
			benchmark_scan(context, scan, datasetSizes[i]);
			delete scan;
		}
	}

	//scan = new clppScan_Merrill(context, datasetSize);
	//benchmark_scan(context, scan);
}

#pragma endregion

#pragma region test_Sort

void test_Sort(clppContext* context)
{
	//---- Brute force
	/*cout << "--------------- Brute force sort" << endl;
	for(unsigned int i = 0; i < datasetSizesCount; i++)
	{
		clppSort* clppsort = new clppSort_CPU(context);
		benchmark_sort(*context, clppsort, datasetSizes[i], PARAM_SORT_BITS);
		delete clppsort;
	}*/

	//---- Radix-sort : generic version
	if (context->isCPU)
	{
		cout << "--------------- Radix sort Key" << endl;
		for(unsigned int i = 0; i < datasetSizesCount; i++)
		{
			clppSort* clppsort = new clppSort_RadixSort(context, datasetSizes[i], PARAM_SORT_BITS, true);
			benchmark_sort(*context, clppsort, datasetSizes[i], PARAM_SORT_BITS);
			delete clppsort;
		}
	}

	//---- Radix-sort : Satish version
	if (context->isGPU)
	{
		cout << "--------------- Satish radix sort Key" << endl;
		for(unsigned int i = 0; i < datasetSizesCount; i++)
		{
			clppSort* clppsort = new clppSort_RadixSortGPU(context, datasetSizes[i], PARAM_SORT_BITS, true);
			benchmark_sort(*context, clppsort, datasetSizes[i], PARAM_SORT_BITS);
			delete clppsort;
		}
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
	//---- Satish Radix-sort
	if (context->isCPU)
	{
		cout << "--------------- Satish sort Key-Value" << endl;
		for(unsigned int i = 0; i < datasetSizesCount; i++)
		{
			clppSort* clppsort = new clppSort_RadixSort(context, datasetSizes[i], PARAM_SORT_BITS, false);
			benchmark_sort_KV(*context, clppsort, datasetSizes[i], PARAM_SORT_BITS);
			delete clppsort;
		}
	}

	//---- Satish Radix-sort
	if (context->isGPU)
	{
		cout << "--------------- Satish sort Key-Value" << endl;
		for(unsigned int i = 0; i < datasetSizesCount; i++)
		{
			clppSort* clppsort = new clppSort_RadixSortGPU(context, datasetSizes[i], PARAM_SORT_BITS, false);
			benchmark_sort_KV(*context, clppsort, datasetSizes[i], PARAM_SORT_BITS);
			delete clppsort;
		}
	}
}

#pragma endregion

#pragma region test_Count

void test_Count(clppContext* context)
{
	int datasetSize = 100000;

	clppCount* counter = new clppCount(context, sizeof(int), 4, datasetSize);

	//---- Prepare the datas-set
	int* values = (int*)malloc(datasetSize * sizeof(int));
	int possiblesValues[] = {1, 2, 4, 8, 16};
	for(int i = 0; i < datasetSize; i++)
	{
		values[i] = possiblesValues[(rand() % 5)];
	}

	counter->pushDatas(values, datasetSize);
	counter->count();

	//---- Check the results
}

#pragma endregion

#pragma region benchmark_scan

void benchmark_scan(clppContext* context, clppScan* scan, int datasetSize)
{
	//---- Create a set of data
	unsigned int* values = (unsigned int*)malloc(datasetSize * sizeof(int));
	//makeOneVector(values, datasetSize);
	makeRandomInt32Vector(values, datasetSize, 8, true);

	//---- Scan : default
	unsigned int* cpuScanValues = (unsigned int*)malloc(datasetSize * sizeof(int));
	memcpy(cpuScanValues, values, datasetSize * sizeof(int));
	cpuScanValues[0] = 0;
	for(unsigned int i = 1; i < datasetSize; i++)
		cpuScanValues[i] = cpuScanValues[i-1] + values[i - 1];

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

	cout << "Performance for data-set size[" << datasetSize << "] time (ms): " << stopWatcher->GetElapsedTime() << endl;

	//---- Free
	free(values);
	free(cpuScanValues);
}

#pragma endregion

#pragma region benchmark_sort

void benchmark_sort(clppContext context, clppSort* sort, unsigned int datasetSize, unsigned int bits)
{
	//---- Create a new set of random datas
	unsigned int* keys = (unsigned int*)malloc(datasetSize * sizeof(int));

	for(unsigned int i = 0; i < PARAM_BENCHMARK_LOOPS; i++)
	{
		makeRandomInt32Vector(keys, datasetSize, bits, true);  

		//---- Push the datas
 		sort->pushDatas(keys, datasetSize);

		//---- Sort
		stopWatcher->StartTimer();
		
		sort->sort();
		sort->waitCompletion();	
		
		stopWatcher->StopTimer();

		//---- Check if it is sorted
		sort->popDatas();
		checkIsSorted(keys, datasetSize, sort->getName(), true, i);
	}

	float time = stopWatcher->GetElapsedTime() / PARAM_BENCHMARK_LOOPS;
	float kps = (1000 / time) * datasetSize;
	cout << "Performance for data-set size[" << datasetSize << "] time (ms): " << time << " KPS[" << (int)kps << "]" << endl;

	//---- Free
	free(keys);
}

#pragma endregion

#pragma region benchmark_sort_KV

void benchmark_sort_KV(clppContext context, clppSort* sort, unsigned int datasetSize, unsigned int bits)
{
	unsigned int* unsortedDatas = (unsigned int*)malloc(2 * datasetSize * sizeof(int));
	unsigned int* unsortedDatasCopy = (unsigned int*)malloc(2 * datasetSize * sizeof(int));

	for(unsigned int i = 0; i < PARAM_BENCHMARK_LOOPS; i++)
	{
		makeRandomInt32Vector(unsortedDatas, datasetSize, bits, false);
		memcpy(unsortedDatasCopy, unsortedDatas, 2 * datasetSize * sizeof(int));

		//---- Push the datas
 		sort->pushDatas(unsortedDatas, datasetSize);

		//---- Sort
		stopWatcher->StartTimer();
		sort->sort();
		sort->waitCompletion();
		stopWatcher->StopTimer();

		float time = stopWatcher->GetElapsedTime() / PARAM_BENCHMARK_LOOPS;
		float kps = (1000 / time) * datasetSize;
		if (i == PARAM_BENCHMARK_LOOPS-1)
			cout << "Performance for data-set size[" << datasetSize << "] time (ms): " << time << " KPS[" << (int)kps << "]" << endl;

		//---- Check if it is sorted
		sort->popDatas();
		checkIsSorted(unsortedDatas, datasetSize, sort->getName(), false, i);
#if PARAM_CHECK_HASLOOSEDVALUES
		checkHasLooseDatasKV(unsortedDatasCopy, unsortedDatas, datasetSize, sort->getName());
#endif
	}

	//---- Free
	free(unsortedDatas);
}

#pragma endregion

#pragma region make...

void makeOneVector(unsigned int* a, unsigned int numElements)
{
    for(unsigned int i = 0; i < numElements; ++i)   
        a[i] = 1; 
}
//
//void makeRandomInt16Vector(unsigned short *a, unsigned int numElements, unsigned int keybits)
//{
//    // Fill up with some random data
//    int keyshiftmask = 0;
//    if (keybits > 16) keyshiftmask = (1 << (keybits - 16)) - 1;
//    int keymask = 0xffff;
//    if (keybits < 16) keymask = (1 << keybits) - 1;
//
//    srand(0);
//    for(unsigned int i = 0; i < numElements; ++i)   
//        a[i] = ((rand() & keyshiftmask)<<16); 
//}
//
//void makeRandomInt32Vector_i(unsigned int* a, unsigned int numElements, unsigned int keybits)
//{
//    for(unsigned int i = 0; i < numElements; ++i)   
//        a[i] = numElements - i; 
//}

//void makeRandomInt32Vector(unsigned int* a, unsigned int numElements, unsigned int keybits)
//{
//    // Fill up with some random data
//    //int keyshiftmask = 0;
//    //if (keybits > 16) keyshiftmask = (1 << (keybits - 16)) - 1;
//    //int keymask = 0xffff;
//    //if (keybits < 16) keymask = (1 << keybits) - 1;
//
//    srand(95123);
//    //cout << "Warning, max int = "<< (1<<_TOTALBITS)<<endl;
//	unsigned int max = (1<<keybits-2) - 1; // Max 'signed' value
//	for(unsigned int i=0; i < numElements; ++i)  { 
//		//a[i] = ((rand() & keyshiftmask)<<16) | (rand() & keymask); 
//		a[i] = rand() % max;
//		//a[i] = i+1;
//		//a[i] = 1;
//    }
//}

template <class T>
void makeShuffleVector(T* a, size_t numElements, unsigned int keybits, bool keysOnly){

	int mult = keysOnly ? 1:2;

	T swap;

	srand(0);
	for(size_t i=0;i< mult*numElements; i++){
		a[i]=i;
	}
	for(size_t i=0;i< mult*numElements; i++){

			swap = a[i];
			size_t j= rand()% (mult*numElements-i);
			a[i]=a[j];
			a[j]=swap;
		}
};

void makeRandomInt32Vector(unsigned int* a, unsigned int numElements, unsigned int keybits, bool keysOnly)
{
	int mult = keysOnly ? 1 : 2;

	unsigned int possiblesValues[] = {1, 2, 4, 8, 16};

	//keybits -= 8;

	srand(0);
	unsigned int max = ( 1 << (keybits-1) ) - 1; // Max 'signed' value	

	//max = ( 1 << (24) ) - 1;
    for(unsigned int i = 0; i < numElements; i++)
	{
		//float rnd = ((double)rand() / (double)RAND_MAX);
		a[i * mult + 0] =  ((double)rand() / (double)RAND_MAX) * (max - 1);
		//a[i * mult + 0] = i;
		//a[i * mult + 0] = numElements+1-i;
		//a[i * mult + 0] = possiblesValues[(rand() % 5)];
		//a[i * mult + 0] = 512-i%512; // to test local sort

		//a[i * mult + 0] &= 0x7FFFFFFF; // To insure it is a signed value

		if (a[i * mult + 0] >= max)
			cout << "Error, max int = " << max << endl;

		if (!keysOnly)
			a[i * mult + 1] = i;
    }
}

// NVidia version
//void makeRandomInt32Vector(unsigned int *a, unsigned int numElements, unsigned int keybits, bool keysOnly)
//{
//	int mult = keysOnly ? 1 : 2;
//
//    // Fill up with some random data
//    int keyshiftmask = 0;
//    if (keybits > 16) keyshiftmask = (1 << (keybits - 16)) - 1;
//    int keymask = 0xffff;
//    if (keybits < 16) keymask = (1 << keybits) - 1;
//
//    srand(95123);
//    for(unsigned int i=0; i < numElements; ++i)   
//    { 
//        a[i * mult + 0] = ((rand() & keyshiftmask)<<16) | (rand() & keymask);
//		a[i * mult + 0] &= 0x7FFFFFFF; // To insure it is a signed value
//
//		if (!keysOnly)
//			a[i * mult + 1] = i;
//    }
//}

#pragma endregion

#pragma region checkIsSorted

bool checkIsSorted(unsigned int* tocheck, size_t datasetSize, string algorithmName, bool keysOnly, int sortId)
{
	int mult = keysOnly ? 1 : 2;
	unsigned int previous = 0;
	for(size_t i = 0; i < datasetSize; i++)
	{
		if (previous > tocheck[i*mult])
		{
			cout << "Algorithm FAILED : LoopId[" << sortId << "] " << algorithmName << endl;
			return false;
		}
		previous = tocheck[i*mult];
	}

	//cout << "SUCCESS : " << algorithmName << endl;

	return true;
}

bool checkHasLooseDatasKV(int* unsorted, int* sorted, size_t datasetSize, string algorithmName)
{
	for(size_t i = 0; i < datasetSize; i++)
	{
		int key = unsorted[i * 2];
		int value = unsorted[i * 2 + 1];
		bool hasFound = false;
		for(size_t j = 0; j < datasetSize; j++)
		{
			int sortedKey = sorted[j * 2];
			int sortedValue = sorted[j * 2 + 1];			

			hasFound = (key == sortedKey && value == sortedValue);
			if (hasFound)	
				break;
		}

		if (!hasFound)
		{
			cout << "Algorithm FAILED, we have loose some datas : " << algorithmName << endl;
			return false;
		}
	}

	return true;
}

#pragma endregion
