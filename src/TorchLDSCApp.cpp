#include <iostream>
#include <iomanip>
#include <vector>

#include "utils/utils.hpp"
#include "./TorchLDSCApp.hpp"

#include <MetNum/Matriz/CImagem.h>

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

    CImagem imageFromStorage("../data/Berea500.pgm");
    EImageType imageType = imageFromStorage.GetFormato();

    int nx = imageFromStorage.NX();
    int ny = imageFromStorage.NY();
    std:vector<int> linearizedData2D;
    RockImageRGB rockImage(model);

    auto tensor = torch::zeros({nx*ny, 1});

    int pos = 0;
    for (auto &&row : imageFromStorage.Data2D())
        for (auto &&value : row) {
            tensor[pos] = rockImage.applyModel(torch::from_blob(std::vector<int>({value, value, value}).data(), {1, 3}, torch::kFloat));
            pos++;
        }

    tensor = tensor.reshape({nx, ny});

    std::shared_ptr<CImagem> output = std::make_shared<CImagem>(nx, ny);

    for (int i = 0; i < tensor.sizes()[0]; i++)
    {
        for (int j = 0; j < tensor.sizes()[1]; j++)
        {   
            output->data2D[i][j] = tensor[i][j].item<int>();
        }
    }
    
    
    output->SetFormato(EImageType::P1_X_Y_ASCII);

    std::ofstream outFile("../data/out", std::ofstream::out);
    output->SalvaDados(outFile); 
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