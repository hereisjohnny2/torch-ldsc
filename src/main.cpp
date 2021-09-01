#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <torch/torch.h>

std::string readDataFromFile(const std::string &filename) {
    std::ifstream file(filename);    
    file.open(filename);    

    if (!file.is_open()) {
       throw std::invalid_argument("Coundn't find the file");  
    }

    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

torch::Tensor fromStringToTensor(const std::string &data) {
    std::vector<double> dataVector = {};
    std::stringstream stream(data);
    std::string tmp;

    while (stream >> tmp)
        dataVector.push_back(std::stod(tmp));

    auto opts = torch::TensorOptions().dtype(torch::kDouble);
    torch::Tensor output = torch::from_blob(dataVector.data(), {int(dataVector.size())/4, 4}, opts).to(torch::kDouble);
    
    return output;
}

int main(int argc, const char** argv) {
    printf("Pytorch and Libldsc integration\n");
    printf("Read file local file...\n");

    std::string filename = "/home/joao/Documentos/dev/C++/test-pytorch/data/training.dat";
    std::string trainingData = readDataFromFile(filename);

    torch::Tensor tensorData = fromStringToTensor(trainingData);

    std::cout << tensorData << std::endl;

    return 0;
}