#include <iostream>
#include <iomanip>

#include "functions.h"

using namespace std;

int main(int argc, char *argv[])
{

    const string VERSION = "0.3.10";

    if (argc > 1)
    {
        if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help") && argc > 1)
        {
            cout << "Summary: bin2cpp [OPTION]... [NAME]" << endl;
            cout << "Example:" << endl << endl;
            cout << "  bin2cpp Data.bin HeaderData DataCopy" << endl << endl;
            cout << "WARNING : Without '.h' extension." << endl << endl;
            return 0;
        }
    }

    if (argc < 3 || argc > 4)
    {
        cout << "Error: Wrong parameters. Use 'bin2cpp -h' for help!" << endl;
        return 1;
    }

    ifstream infile (argv[1], ios_base::in | ios_base::binary);

    if (!infile)
    {
        cout << "Error: Unable to find the binary file!" << endl;
        return 1;
    }

    stringstream headername, structname_buffer;
    string structname;

    structname_buffer << argv[2];
    structname = strtoupper(structname_buffer.str());

    headername << argv[2] << ".h";

    ofstream outfile (headername.str().c_str());

    if (!outfile)
    {
        cout << "Error: Unable to create the output header file!" << endl;
        outfile.close();

        return 1;
    }

    long int ifilesize = filesize(argv[1]);

    outfile << "const unsigned int " << structname << "_LENGTH = " << ifilesize << endl;

    // if (argc < 4)
        // outfile << "const string " << structname << "_NAME \"" << argv[1] << "\"" << endl;
    // else
        // outfile << "const string " << structname << "_NAME \"" << argv[3] << "\"" << endl;

    outfile << "\nconst unsigned char " << structname << "[]=\n{\n  ";

    char buff[1];
    int b = 1;
    int bar_size = ifilesize / 50;
    int ic = 0;
    string bar = "[                                                  ]";

    cout << "Progression : " << bar << " 0%" << flush;

    while (!infile.eof())
    {
        infile.read(buff, 1);

        if (ic != 0 && ic % 30 == 0)
                outfile << endl << "  ";

        if ((ic+1) % bar_size == 0)
        {
            bar[b] = '=';
            cout << "\rProgression : " << bar << " " << (b*2) << "%" << flush;
            ++b;
        }

        if (!infile.eof())
            outfile << "0x" << hex << setfill('0')<<  setw(2) << static_cast<int>(*buff & 0xFF) << ",";
        else
            outfile << "0x" << hex << setfill('0')<<  setw(2) << static_cast<int>(*buff & 0xFF);

        ic++;
    }

    outfile << "\n};" << endl;

    infile.close();
    outfile.close();

    cout << "\n\"" << argv[1] << "\" Completed in " << setprecision(2) << fixed << " : \"" << headername.str() << endl;

    return 0;
}