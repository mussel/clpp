#ifndef __CLPP_SCAN_H__
#define __CLPP_SCAN_H__

#include "clpp/clppProgram.h"

class clppScan : public clppProgram
{
public:
	virtual void scan() = 0;

	virtual void pushDatas(void* values, void* valuesOut, size_t valueSize, size_t datasetSize) = 0;
	virtual void pushDatas(cl_mem clBuffer_values, cl_mem clBuffer_valuesOut, size_t valueSize, size_t datasetSize) = 0;

	virtual void popDatas() = 0;

protected:
	size_t _datasetSize;	// The number of values to scan

	void* _values;			// The associated data set to scan
	void* _valuesOut;		// The scanned data set
	size_t _valueSize;		// The size of a value in bytes

	cl_mem _clBuffer_values;
	cl_mem _clBuffer_valuesOut;

	size_t _workgroupSize;
};

#endif