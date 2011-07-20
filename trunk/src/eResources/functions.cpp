#include "functions.h"

#include <algorithm>
#include <cctype>       // std::toupper
#include <string>

long filesize(char *fp)
{
    ifstream infilesize(fp, ios_base::in);

    if (infilesize)
    {
        infilesize.seekg(0L, ios_base::end);

        return infilesize.tellg();
    }

	return 1;
}

string strtoupper(string s)
{
    transform(s.begin(), s.end(), s.begin(), static_cast<int(*)(int)>(toupper));

    return s;
}

