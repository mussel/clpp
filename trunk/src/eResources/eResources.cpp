#include "stdafx.h"

#include <iostream>
#include <iomanip>

#include "functions.h"

using namespace std;

// Optimizations :
// - remove emty lines
// - trim
// - remove comments

void Trim(string& str);

int _tmain(int argc, char* argv[])
{
	string inputFile = argv[1];
	inputFile += ".cl";

	string headername = argv[1];
	headername = headername + "_CLKernel.h";

	string structname = argv[1];
	structname = "clCode_" + structname;

	ifstream infile(inputFile.c_str(), ios_base::in | ios_base::binary);
    if (!infile)
    {
        cout << "Error: Unable to find the binary file!" << endl;
        return 1;
    }
    
    ofstream outfile (headername.c_str());

    if (!outfile)
    {
        cout << "Error: Unable to create the output header file!" << endl;
        outfile.close();

        return 1;
    }

    outfile << "\nchar " << structname << "[]=\n";

	char buffer[5000];
	char newline = '\n';
	while(!infile.eof())
	{
		infile.getline(buffer, 5000);

		// Check if we need to encode this line !
		string text(buffer);
		Trim(text);
		if (text.length() < 1 || (text.length() > 1 && text.substr(0,2) == "//"))
			continue;

		if (text.length() == 1 && text[0] == 13)
			continue;

		outfile << "\"";

		int len = text.length();
		for(int i = 0; i < len; i++)
			if (text[i] != 13)
				outfile << text[i];

		outfile << "\\n\"" << endl;
	}

    outfile << ";" << endl;


    infile.close();
    outfile.close();

    return 0;
}

void Trim(string& str)
{
	string::size_type pos = str.find_last_not_of(' ');
	if(pos != string::npos)
	{
		str.erase(pos + 1);
		pos = str.find_first_not_of(' ');
		if(pos != string::npos) str.erase(0, pos);
	}
	else str.erase(str.begin(), str.end());
}