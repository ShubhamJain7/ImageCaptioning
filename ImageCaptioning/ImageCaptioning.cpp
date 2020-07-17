#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

int main()
{
    vector<string> vocab;
    ifstream file("C:/Users/dell/source/repos/ImageCaptioning/data/vocab.txt");
    string line;
    while (std::getline(file, line)) {
        vocab.push_back(line);
    }
    for (size_t i = 0; i < 10; i++)
    {
        cout << vocab[i] << endl;
    }
}