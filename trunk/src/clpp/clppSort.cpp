#include "clpp/clppSort.h"

string clppSort::loadKernelSource(string path)
{
	string kernel = "";
	char buffer[5000];
	ifstream infile(path.c_str(), ios_base::in | ios_base::binary);

	if (!infile)
        return "";

	while(!infile.eof())
	{
		infile.getline(buffer, 5000);

		string text(buffer);
		kernel += text + "\n";
	}
	infile.close();

	return kernel;
}