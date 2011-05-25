#include "clpp/clppSort_Blelloch.h"

clppSort_Blelloch::clppSort_Blelloch(clppContext* context, string basePath)
{
	_context = context;

	nkeys = _N;
	nkeys_rounded = _N;

	//---- Read the source code
	_kernelSource = loadKernelSource(basePath + "clppSort_Blelloch.cl");

	cl_int clStatus;
	const char* ptr = _kernelSource.c_str();
	size_t len = _kernelSource.length();

	//---- Build the program
	clProgram = clCreateProgramWithSource(context->clContext, 1, (const char **)&ptr, &len, &clStatus);
	checkCLStatus(clStatus);

	clStatus = clBuildProgram(clProgram, 0, NULL, NULL, NULL, NULL);
  
	if (clStatus != CL_SUCCESS)
	{
		size_t len;
		char buffer[5000];
		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(clProgram, context->clDevice, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);

		//printf("%s\n", buffer);
		printf("%s\n", getOpenCLErrorString(clStatus));

		checkCLStatus(clStatus);
	}

	//---- Prepare all the kernels
	kernel_Histogram = clCreateKernel(clProgram, "histogram", &clStatus);
	checkCLStatus(clStatus);
	
	kernel_ScanHistogram = clCreateKernel(clProgram, "scanhistograms", &clStatus);
	checkCLStatus(clStatus);

	kernel_PasteHistogram = clCreateKernel(clProgram, "pastehistograms", &clStatus);
	checkCLStatus(clStatus);

	kernel_Reorder = clCreateKernel(clProgram, "reorder", &clStatus);
	checkCLStatus(clStatus);

	kernel_Transpose = clCreateKernel(clProgram, "transpose", &clStatus);
	checkCLStatus(clStatus);
}

void clppSort_Blelloch::sort(void* keys, void* values, size_t datasetSize, unsigned int keyBits)
{
	//---- Send the data to the devices
	initializeCLBuffers(keys, values, datasetSize);

	//---- We set here the fixed arguments of the OpenCL kernels
	// the changing arguments are modified elsewhere in the class
	cl_int clStatus;
	clStatus = clSetKernelArg(kernel_Histogram, 1, sizeof(cl_mem), &_clBuffer_Histograms);
	checkCLStatus(clStatus);

	clStatus = clSetKernelArg(kernel_Histogram, 3, sizeof(int)*_RADIX*_ITEMS, NULL);
	checkCLStatus(clStatus);

	// clStatus = clSetKernelArg(kernel_Histogram, 3, sizeof(int)*_ITEMS, NULL);
	// checkCLStatus(clStatus);

	clStatus = clSetKernelArg(kernel_PasteHistogram, 0, sizeof(cl_mem), &_clBuffer_Histograms);
	checkCLStatus(clStatus);

	clStatus = clSetKernelArg(kernel_PasteHistogram, 1, sizeof(cl_mem), &_clBuffer_globsum);
	checkCLStatus(clStatus);

	clStatus = clSetKernelArg(kernel_Reorder, 2, sizeof(cl_mem), &_clBuffer_Histograms);
	checkCLStatus(clStatus);

	clStatus  = clSetKernelArg(kernel_Reorder, 6, sizeof(int) * _RADIX * _ITEMS , NULL); // mem cache
	checkCLStatus(clStatus);

	//---- Sort
	Sort();

	//---- Retreive the data from the devices
}

void clppSort_Blelloch::initializeCLBuffers(void* keys, void* values, size_t datasetSize)
{
	cl_int clStatus;

	//---- Construction of the initial permutation
	for(size_t i = 0; i < datasetSize; i++)
		_permutations[i] = i;

	//---- Create all the buffers
	_clBuffer_inKeys  = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, sizeof(int)* datasetSize, NULL, &clStatus);
	checkCLStatus(clStatus);

	_clBuffer_outKeys  = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, sizeof(int)* datasetSize, NULL, &clStatus);
	checkCLStatus(clStatus);

	_clBuffer_inPermutations = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, sizeof(int)* datasetSize, NULL, &clStatus);
	checkCLStatus(clStatus);

	_clBuffer_outPermutations = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, sizeof(int)* datasetSize, NULL, &clStatus);
	checkCLStatus(clStatus);

	// copy on the GPU
	_clBuffer_Histograms  = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, sizeof(int)* _RADIX * _GROUPS * _ITEMS, NULL, &clStatus);
	checkCLStatus(clStatus);

	// copy on the GPU
	_clBuffer_globsum  = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, sizeof(int)* _HISTOSPLIT, NULL, &clStatus);
	checkCLStatus(clStatus);

	// temporary vector when the sum is not needed
	_clBuffer_temp  = clCreateBuffer(_context->clContext, CL_MEM_READ_WRITE, sizeof(int)* _HISTOSPLIT, NULL, &clStatus);
	checkCLStatus(clStatus);

	Resize(nkeys);

	//---- Send the data
	clStatus = clEnqueueWriteBuffer(_context->clQueue, _clBuffer_inKeys, CL_FALSE, 0, sizeof(int) * datasetSize, keys, 0, NULL, NULL);
	checkCLStatus(clStatus);

	clStatus = clEnqueueWriteBuffer(_context->clQueue, _clBuffer_inPermutations, CL_FALSE, 0, sizeof(int) * datasetSize, _permutations, 0, NULL, NULL);
	checkCLStatus(clStatus);
}

// resize the sorted vector
void clppSort_Blelloch::Resize(int nn)
{
    nkeys = nn;

    // length of the vector has to be divisible by (_GROUPS * _ITEMS)
    int remainder = nkeys % (_GROUPS * _ITEMS);
    nkeys_rounded = nkeys;
    cl_int clStatus;
    unsigned int pad[_GROUPS * _ITEMS];
    for (int ii=0;ii<_GROUPS * _ITEMS;ii++)
        pad[ii]=_MAXINT - (unsigned int)1;

    if (remainder != 0)
    {
        nkeys_rounded = nkeys - remainder + (_GROUPS * _ITEMS);

        // pad the vector with big values
        assert(nkeys_rounded <= _N);
        clStatus = clEnqueueWriteBuffer(_context->clQueue, _clBuffer_inKeys, CL_TRUE, sizeof(int) * nkeys, sizeof(int) * (_GROUPS * _ITEMS - remainder), pad, 0, NULL, NULL);
        
        checkCLStatus(clStatus);
    }
}

void clppSort_Blelloch::Transpose(int nbrow,int nbcol)
{
    cl_int clStatus;

    clStatus  = clSetKernelArg(kernel_Transpose, 0, sizeof(cl_mem), &_clBuffer_inKeys);
    checkCLStatus(clStatus);

    clStatus  = clSetKernelArg(kernel_Transpose, 1, sizeof(cl_mem), &_clBuffer_outKeys);
    checkCLStatus(clStatus);

    clStatus = clSetKernelArg(kernel_Transpose, 2, sizeof(int), &nbcol);
    checkCLStatus(clStatus);

    clStatus = clSetKernelArg(kernel_Transpose, 3, sizeof(int), &nbrow);
    checkCLStatus(clStatus);

    clStatus  = clSetKernelArg(kernel_Transpose, 4, sizeof(cl_mem), &_clBuffer_inPermutations);
    checkCLStatus(clStatus);

    clStatus  = clSetKernelArg(kernel_Transpose, 5, sizeof(cl_mem), &_clBuffer_outPermutations);
    checkCLStatus(clStatus);

    clStatus  = clSetKernelArg(kernel_Transpose, 6, sizeof(int)*_GROUPS*_GROUPS, NULL);
    checkCLStatus(clStatus);

    clStatus  = clSetKernelArg(kernel_Transpose, 7, sizeof(int)*_GROUPS*_GROUPS, NULL);
    checkCLStatus(clStatus);

    cl_event eve;
    size_t global_work_size[2];
    size_t local_work_size[2];

    assert(nbrow%_GROUPS == 0);
    assert(nbcol%_GROUPS == 0);

    global_work_size[0]=nbrow/_GROUPS;
    global_work_size[1]=nbcol;

    local_work_size[0]=1;
    local_work_size[1]=_GROUPS;

	// two dimensions: rows and columns
	clStatus = clEnqueueNDRangeKernel(_context->clQueue, kernel_Transpose, 2, NULL, global_work_size, local_work_size, 0, NULL, &eve);
	checkCLStatus(clStatus);

    // exchange the pointers

    // swap the old and new vectors of keys
    cl_mem _clBuffer_temp;
    _clBuffer_temp = _clBuffer_inKeys;
    _clBuffer_inKeys = _clBuffer_outKeys;
    _clBuffer_outKeys = _clBuffer_temp;

    // swap the old and new permutations
    _clBuffer_temp = _clBuffer_inPermutations;
    _clBuffer_inPermutations = _clBuffer_outPermutations;
    _clBuffer_outPermutations = _clBuffer_temp;

    // timing
    clFinish(_context->clQueue);

    cl_ulong beginning, end;

    clStatus = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), (void*)&beginning, NULL);
    checkCLStatus(clStatus);

    clStatus = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), (void*)&end, NULL);
    checkCLStatus(clStatus);

	_timerTranspose += (float)(end-beginning)/1e9;
}

void clppSort_Blelloch::Sort()
{
    //assert(nkeys_rounded <= _N);
    //assert(nkeys <= nkeys_rounded);

    int nbcol = nkeys_rounded / (_GROUPS * _ITEMS);
    int nbrow = _GROUPS * _ITEMS;

    if (TRANSPOSE)
		Transpose(nbrow, nbcol);

    for(unsigned int pass = 0; pass < _PASS; pass++)
    {
        Histogram(pass);
        ScanHistogram();
        Reorder(pass);
    }

    if (TRANSPOSE)
        Transpose(nbcol, nbrow);

    _timerSort = _timerHisto + _timerScan + _timerReorder + _timerTranspose;
}

void clppSort_Blelloch::Histogram(int pass)
{
    cl_int clStatus;

    size_t nblocitems=_ITEMS;
    size_t nbitems=_GROUPS*_ITEMS;

    assert(_RADIX == pow(2.f,_BITS));

    clStatus  = clSetKernelArg(kernel_Histogram, 0, sizeof(cl_mem), &_clBuffer_inKeys);
    checkCLStatus(clStatus);

    clStatus = clSetKernelArg(kernel_Histogram, 2, sizeof(int), &pass);
    checkCLStatus(clStatus);

    assert(nkeys_rounded%(_GROUPS * _ITEMS) == 0);
    assert(nkeys_rounded <= _N);

    clStatus = clSetKernelArg(kernel_Histogram, 4, sizeof(int), &nkeys_rounded);
    checkCLStatus(clStatus);

    cl_event eve;

    clStatus = clEnqueueNDRangeKernel(_context->clQueue, kernel_Histogram, 1, NULL, &nbitems, &nblocitems, 0, NULL, &eve);

    //cout << clStatus<<" , "<<CL_OUT_OF_RESOURCES<<endl;
    checkCLStatus(clStatus);

    clFinish(_context->clQueue);

    cl_ulong beginning,end;

    clStatus = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), (void*) &beginning, NULL);
    checkCLStatus(clStatus);

    clStatus = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), (void*) &end, NULL);
    checkCLStatus(clStatus);

    _timerHisto += (float)(end-beginning)/1e9;
}

void clppSort_Blelloch::ScanHistogram()
{
    cl_int clStatus;

    // numbers of processors for the local scan
    // half the size of the local histograms
    size_t nbitems = _RADIX* _GROUPS*_ITEMS / 2;

    size_t nblocitems = nbitems/_HISTOSPLIT ;

    int maxmemcache=max(_HISTOSPLIT,_ITEMS * _GROUPS * _RADIX / _HISTOSPLIT);

    // scan locally the histogram (the histogram is split into several
    // parts that fit into the local memory)

    clStatus = clSetKernelArg(kernel_ScanHistogram, 0, sizeof(cl_mem), &_clBuffer_Histograms);
    checkCLStatus(clStatus);

    clStatus  = clSetKernelArg(kernel_ScanHistogram, 1, sizeof(int)* maxmemcache , NULL); // mem cache

    clStatus = clSetKernelArg(kernel_ScanHistogram, 2, sizeof(cl_mem), &_clBuffer_globsum);
    checkCLStatus(clStatus);

    cl_event eve;
    clStatus = clEnqueueNDRangeKernel(_context->clQueue, kernel_ScanHistogram, 1, NULL, &nbitems, &nblocitems, 0, NULL, &eve);

    // cout << clStatus<<","<< CL_INVALID_WORK_ITEM_SIZE<< " "<<nbitems<<" "<<nblocitems<<endl;
    // cout <<CL_DEVICE_MAX_WORK_ITEM_SIZES<<endl;
    checkCLStatus(clStatus);
    clFinish(_context->clQueue);

    cl_ulong beginning,end;

    clStatus = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), (void*)&beginning, NULL);
    checkCLStatus(clStatus);

    clStatus = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), (void*)&end, NULL);
    checkCLStatus(clStatus);

    _timerScan += (float)(end-beginning)/1e9;

    // second scan for the globsum
    clStatus = clSetKernelArg(kernel_ScanHistogram, 0, sizeof(cl_mem), &_clBuffer_globsum);
    checkCLStatus(clStatus);

    clStatus = clSetKernelArg(kernel_ScanHistogram, 2, sizeof(cl_mem), &_clBuffer_temp);
    checkCLStatus(clStatus);

    nbitems= _HISTOSPLIT / 2;
    nblocitems=nbitems;

    clStatus = clEnqueueNDRangeKernel(_context->clQueue, kernel_ScanHistogram, 1, NULL, &nbitems, &nblocitems, 0, NULL, &eve);

    checkCLStatus(clStatus);
    clFinish(_context->clQueue);

    clStatus = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), (void*) &beginning, NULL);
    checkCLStatus(clStatus);

    clStatus = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), (void*)&end, NULL);
    checkCLStatus(clStatus);

    _timerScan += (float)(end-beginning)/1e9;

	// loops again in order to paste together the local histograms
    nbitems = _RADIX* _GROUPS*_ITEMS/2;
    nblocitems=nbitems/_HISTOSPLIT;

    clStatus = clEnqueueNDRangeKernel(_context->clQueue, kernel_PasteHistogram, 1, NULL, &nbitems, &nblocitems, 0, NULL, &eve);

    checkCLStatus(clStatus);
    clFinish(_context->clQueue);

    clStatus = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), (void*)&beginning, NULL);
    checkCLStatus(clStatus);

    clStatus = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), (void*)&end, NULL);
    checkCLStatus(clStatus);

    _timerScan += (float)(end-beginning)/1e9;
}
/*
// chekernel_ the computation at the end
void clppSort_Blelloch::Chekernel_()
{

    cout << "Get the data from the GPU"<<endl;

    RecupGPU();

    cout << "Test order"<<endl;

    // first see if the final list is ordered
    for (int i=0;i<nkeys-1;i++)
    {
        if (!(h_Keys[i] <= h_Keys[i+1]))
        {
            cout <<"erreur tri "<< i<<" "<<h_Keys[i]<<" ,"<<i+1<<" "<<h_Keys[i+1]<<endl;
        }
        assert(h_Keys[i] <= h_Keys[i+1]);
    }

    if (PERMUT)
    {
        cout << "Chekernel_ the permutation"<<endl;
        // chekernel_ if the permutation corresponds to the original list
        for (int i=0;i<nkeys;i++)
        {
            if (!(h_Keys[i] == h_chekernel_Keys[h_Permut[i]]))
            {
                cout <<"erreur permut "<< i<<" "<<h_Keys[i]<<" ,"<<i+1<<" "<<h_Keys[i+1]<<endl;
            }
            //assert(h_Keys[i] == h_chekernel_Keys[h_Permut[i]]);
        }
    }

    cout << "test OK !"<<endl;

}
*/

void clppSort_Blelloch::Reorder(int pass)
{
    cl_int clStatus;

    size_t nblocitems=_ITEMS;
    size_t nbitems=_GROUPS*_ITEMS;


    clFinish(_context->clQueue);

    clStatus  = clSetKernelArg(kernel_Reorder, 0, sizeof(cl_mem), &_clBuffer_inKeys);
    checkCLStatus(clStatus);

    clStatus  = clSetKernelArg(kernel_Reorder, 1, sizeof(cl_mem), &_clBuffer_outKeys);
    checkCLStatus(clStatus);

    clStatus = clSetKernelArg(kernel_Reorder, 3, sizeof(int), &pass);
    checkCLStatus(clStatus);

    clStatus  = clSetKernelArg(kernel_Reorder, 4, sizeof(cl_mem), &_clBuffer_inPermutations);
    checkCLStatus(clStatus);

    clStatus  = clSetKernelArg(kernel_Reorder, 5, sizeof(cl_mem), &_clBuffer_outPermutations);
    checkCLStatus(clStatus);

    clStatus  = clSetKernelArg(kernel_Reorder, 6,
                          sizeof(int)* _RADIX * _ITEMS ,
                          NULL); // mem cache
    checkCLStatus(clStatus);

    assert(nkeys_rounded%(_GROUPS * _ITEMS) == 0);

    clStatus = clSetKernelArg(kernel_Reorder, 7, sizeof(int), &nkeys_rounded);
    checkCLStatus(clStatus);


    assert(_RADIX == pow(2.f, _BITS));

    cl_event eve;

    clStatus = clEnqueueNDRangeKernel(_context->clQueue, kernel_Reorder, 1, NULL, &nbitems, &nblocitems, 0, NULL, &eve);

    checkCLStatus(clStatus);
    clFinish(_context->clQueue);

    cl_ulong beginning,end;

    clStatus = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), (void*) &beginning, NULL);
    checkCLStatus(clStatus);

    clStatus = clGetEventProfilingInfo(eve, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), (void*) &end, NULL);
    checkCLStatus(clStatus);

    _timerReorder += (float)(end-beginning)/1e9;

    // swap the old and new vectors of keys
    cl_mem _clBuffer_temp;
    _clBuffer_temp=_clBuffer_inKeys;
    _clBuffer_inKeys=_clBuffer_outKeys;
    _clBuffer_outKeys=_clBuffer_temp;

    // swap the old and new permutations
    _clBuffer_temp=_clBuffer_inPermutations;
    _clBuffer_inPermutations=_clBuffer_outPermutations;
    _clBuffer_outPermutations=_clBuffer_temp;
}
