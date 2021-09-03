#include <iostream>
#include <string>
#include <iomanip>
#include <memory>
#include "utils/utils.hpp"

#include "./TorchLDSCApp.hpp"
#include "RockImageRGB/RockImageRGB.hpp"

void TorchLDSCApp::run() 
{
    std::string data;

    auto model = std::make_shared<RockImageRGBNet>();
    torch::optim::SGD optimizer(model->parameters(), 0.04);
    RockImageRGB rockImage(model);

    std::cout << "LIBLDSC - TORCH Integration" << std::endl;

    std::cout << "\t1 - Train Model" << std::endl;
    std::cout << "\t2 - Run Model" << std::endl;
    std::cout << "Choose on of the options: ";

    int choice;
    std::cin >> choice;
    double rgb[] = {0.165, 0.984, 0.876};
    RGBValueDTO testRgb(rgb);
    int output;

    switch (choice)
    {
    case 1:
        rockImage.train(data, optimizer);
        break;
    case 2:
        output = rockImage.runModel(testRgb);   
        std::cout << "Output: " << output << std::endl;
        break;
    default:
        std::cout << "Invalid Choice!" << std::endl;
        break;
    }   
}