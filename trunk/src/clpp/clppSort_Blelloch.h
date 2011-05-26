#ifndef __CLPP_SORT_BLELLOCH_H__
#define __CLPP_SORT_BLELLOCH_H__

#include<math.h>

#include "clpp/clppSort.h"

// these parameters can be changed
#define _ITEMS  32 // number of items in a group
#define _GROUPS 16 // the number of virtual processors is _ITEMS * _GROUPS
#define  _HISTOSPLIT 128 // number of splits of the histogram
#define _TOTALBITS 16  // number of bits for the integer in the list (max=32)
#define _BITS 4  // number of bits in the radix

// max size of the sorted vector
// it has to be divisible by  _ITEMS * _GROUPS
// (for other sizes, pad the list with big values)
#define _N (_ITEMS * _GROUPS * 16)  
//#define _N (1<<21)  // maximal size of the list  
#define VERBOSE 1
#define TRANSPOSE 1  // transpose the initial vector (faster memory access)
#define PERMUT 1  // store the final permutation

// the following parameters are computed from the previous
#define _RADIX (1 << _BITS) //  radix  = 2^_BITS
#define _PASS (_TOTALBITS/_BITS) // number of needed passes to sort the list
#define _HISTOSIZE (_ITEMS * _GROUPS * _RADIX ) // size of the histogram

// maximal value of integers for the sort to be correct
#define _MAXINT (1 << (_TOTALBITS-1))

class clppSort_Blelloch : public clppSort
{
public:
	clppSort_Blelloch(clppContext* context, string basePath);

	// Returns the algorithm name
	string getName() { return "Blelloch"; }

	// Sort the pushed data set 
	void sort();

	// Push the data on the device
	void pushDatas(void* keys, void* values, size_t valueSize, size_t datasetSize, unsigned int keyBits);

	// Pop the data from the device
	void popDatas();

private:
	string _kernelSource;

	cl_program clProgram;
	cl_kernel kernel_Histogram;
	cl_kernel kernel_ScanHistogram;
	cl_kernel kernel_PasteHistogram;
	cl_kernel kernel_Reorder;
	cl_kernel kernel_Transpose;

	void initializeCLBuffers(void* keys, void* values, size_t datasetSize);

	int _permutations[_N];

	cl_mem _clBuffer_inKeys;
	cl_mem _clBuffer_outKeys;
	cl_mem _clBuffer_inPermutations;
	cl_mem _clBuffer_outPermutations;
	cl_mem _clBuffer_Histograms;
	cl_mem _clBuffer_globsum;
	cl_mem _clBuffer_temp;

	// Resize the sorted vector
	void resize(int nn);

	// Transpose the list for faster memory access
	void transpose(int nbrow, int nbcol);

	void histogram(int pass);

	void scanHistogram();

	// Reorder the data from the scanned histogram
	void reorder(int pass);

	int nkeys;
	int nkeys_rounded;

	// timers
	float _timerHisto;
	float _timerScan;
	float _timerReorder;
	float _timerSort;
	float _timerTranspose;
};

#endif