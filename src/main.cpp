#include <iostream>
#include <torch/torch.h>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

std::string readDataFromFile(const std::string &filename) {
    std::ifstream file(filename);    
    file.open(filename);    

    if (!file.is_open()) {
       throw std::invalid_argument("Coundn't find the file");  
    }

    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

int main(int argc, const char** argv) {
    printf("Pytorch and Libldsc integration\n");
    printf("Read file local file...\n");

    std::string filename = "/home/joao/Documentos/dev/C++/test-pytorch/data/points.dat";
    std::string data = readDataFromFile(filename);

    printf("Data: %s", data.c_str());

    return 0;
}