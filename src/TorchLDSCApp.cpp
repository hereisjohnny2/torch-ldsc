#include <iostream>
#include <iomanip>

#include "utils/utils.hpp"
#include "./TorchLDSCApp.hpp"

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
    std::string data = utils::readDataFromFile("../data/training.dat");

    auto model = std::make_shared<RockImageRGBNet>();
    torch::optim::SGD optimizer(model->parameters(), 0.001);

    RockImageRGB rockImage(model);
    rockImage.train(data, optimizer);

    saveModel(model, "../data/model.pt");
}

void TorchLDSCApp::runModel()
{
    auto model = std::make_shared<RockImageRGBNet>();
    loadModel(model, "../data/model.pt");

    std::cout << "\nSaved model: \n";
    for (auto& p : model->named_parameters()) {
        std::cout <<  p.key() << " - " << p.value() << "\n\n";
    }

    double rgb[] = {0.165, 0.984, 0.876};
    RGBValueDTO testRgb(rgb);

    RockImageRGB rockImage(model);
    
    int output = rockImage.runModel(testRgb);   
    std::cout << "Output: " << output << std::endl;
}

void TorchLDSCApp::showMainMenu()
{
    std::cout << "LIBLDSC - TORCH Integration" << std::endl;

    std::cout << "\t1 - Train Model" << std::endl;
    std::cout << "\t2 - Run Model" << std::endl;
    std::cout << "Choose one of the options: ";
}


void TorchLDSCApp::saveModel(std::shared_ptr<RockImageRGBNet> model, const std::string &filename) {
    std::ifstream file(filename);    
    file.open(filename);    

    if (!file.is_open()) {
        std::cerr <<  "\nCoundn't find the path!\n";
        return;
    }

    file.close();
    
    torch::save(model, filename);
}

void TorchLDSCApp::loadModel(std::shared_ptr<RockImageRGBNet> model, const std::string &filename) {
    std::ifstream file(filename);    
    file.open(filename);    

    if (!file.is_open()) {
        std::cerr <<  "\nCoundn't find the file!\n";
        return;
    }

    file.close();
    
    torch::load(model, filename);
}