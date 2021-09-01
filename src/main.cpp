#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include "Net.hpp"
#include "RockImageRGBDataset.hpp"
#include <memory>

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
    
    torch::Tensor output = torch::zeros({(int)dataVector.size() / 4, 4});
    for (int i = 0; i < dataVector.size() / 4; i++)
    {
        output[i][0] = dataVector[4*i    ];
        output[i][1] = dataVector[4*i + 1];
        output[i][2] = dataVector[4*i + 2];
        output[i][3] = dataVector[4*i + 3];
    }
    
    return output;
}

int main(int argc, const char** argv) {
    std::string filename = "/home/joao/Documentos/dev/C++/test-pytorch/data/training.dat";
    std::string trainingData = readDataFromFile(filename);

    auto dataset = RockImageRGBDataset(trainingData).map(torch::data::transforms::Stack<>());

    auto net = std::make_shared<Net>();
    torch::optim::SGD optimizer(net->parameters(), /*lr = */0.01);

    return 0;
}