#include <iostream>
#include <string>
#include <iomanip>
#include <memory>
#include "utils/utils.hpp"

#include "./TorchLDSCApp.hpp"
#include "RockImageRGB/RockImageRGB.hpp"

void TorchLDSCApp::run() 
{
    showMainMenu();

    int choice;
    std::cin >> choice;
    
    switch (choice)
    {
    case 1:
        runTraining();
        break;
    case 2:
        runModel();
        break;
    default:
        std::cout << "Invalid Choice!" << std::endl;
        break;
    }   
}

void TorchLDSCApp::runTraining()
{
    std::string data;

    auto model = std::make_shared<RockImageRGBNet>();
    torch::optim::SGD optimizer(model->parameters(), 0.04);

    RockImageRGB rockImage(model);
    rockImage.train(data, optimizer);
}

void TorchLDSCApp::runModel()
{
    auto model = std::make_shared<RockImageRGBNet>();
    RockImageRGB rockImage(model);

    double rgb[] = {0.165, 0.984, 0.876};
    RGBValueDTO testRgb(rgb);
    
    int output = rockImage.runModel(testRgb);   
    std::cout << "Output: " << output << std::endl;
}

void TorchLDSCApp::showMainMenu()
{
    std::cout << "LIBLDSC - TORCH Integration" << std::endl;

    std::cout << "\t1 - Train Model" << std::endl;
    std::cout << "\t2 - Run Model" << std::endl;
    std::cout << "Choose on of the options: ";
}
