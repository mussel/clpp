#include "clpp/clppSort_CPU.h"

#include <algorithm>

#pragma region Construsctor

clppSort_CPU::clppSort_CPU(clppContext* context)
{
}

clppSort_CPU::~clppSort_CPU()
{
}

#pragma endregion

#pragma region sort

void clppSort_CPU::sort()
{
	//std::sort((char*)_keys, (char*)_keys + _datasetSize * (_keyBits/8));
	std::sort((int*)_keys, ((int*)_keys) + _datasetSize);
}

#pragma endregion

#pragma region pushDatas

void clppSort_CPU::pushDatas(void* keys, void* values, size_t keySize, size_t valueSize, size_t datasetSize, unsigned int keyBits)
{
	_keys = keys;
	_keySize = keySize;
	_values = values;
	_valueSize = valueSize;
	_datasetSize = datasetSize;
	_keyBits = keyBits;
}

void clppSort_CPU::pushCLDatas(cl_mem clBuffer_keys, cl_mem clBuffer_values, size_t datasetSize, unsigned int keyBits) 
{
	// Unsupported of course !
}

#pragma endregion

#pragma region popDatas

void clppSort_CPU::popDatas()
{
}

#pragma endregion