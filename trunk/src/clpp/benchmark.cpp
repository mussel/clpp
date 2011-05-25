#include <stdlib.h>

#include "clpp/clppSort_Blelloch.h"

void makeRandomUintVector(unsigned int *a, unsigned int numElements, unsigned int keybits);

int main(int argc, const char **argv)
{
	unsigned int datasetSize = 100000;

	//---- Create a new set of random datas
	unsigned int* keys = (unsigned int*)malloc(datasetSize * sizeof(int));
	makeRandomUintVector(keys, datasetSize, 32);

	//---- Prepare a clpp Context
	clppContext context;

	context.setup();

	//---- Blelloch
	clppSort_Blelloch* sort01 = new clppSort_Blelloch(&context, "src/clpp/");
	sort01->sort(keys, keys, datasetSize, 32);

	//---- Free
	free(keys);
}

void makeRandomUintVector(unsigned int *a, unsigned int numElements, unsigned int keybits)
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