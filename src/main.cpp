#include <iostream>
#include <string>
#include <iomanip>
#include <memory>
#include "utils/utils.hpp"

#include <torch/optim.h>
#include "RockImageRGB/RockImageRGB.hpp"


int main(int argc, const char** argv) {
    double lr = 0.04;
    int batch_size = 5;

    std::string filename = "/home/joao/Documentos/dev/C++/test-pytorch/data/training.dat";
    std::string data = utils::readDataFromFile(filename);

    auto model = std::make_shared<RockImageRGBNet>();
    torch::optim::SGD optimizer(model->parameters(), lr);

    RockImageRGB rockImage(model);
    // rockImage.train(data, optimizer);

    double rgb[] = {0.165, 0.984, 0.876};
    RGBValueDTO testRgb(rgb);

    int output = rockImage.runModel(testRgb);
    

    return 0;
}