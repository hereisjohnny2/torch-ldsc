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

int main(int argc, const char** argv) {
    std::string filename = "/home/joao/Documentos/dev/C++/test-pytorch/data/training.dat";
    std::string trainingData = readDataFromFile(filename);

    auto dataset = RockImageRGBDataset(trainingData).map(torch::data::transforms::Stack<>());
    auto dataLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(dataset),
        /*batch_size = */5
    );

    auto net = std::make_shared<Net>();
    torch::optim::SGD optimizer(net->parameters(), /*lr = */0.05);

    auto dataset_size = dataset.size().value();
    for (int epoch = 0; epoch <= 50; epoch++)
    {
        for (auto &batch : *dataLoader)
        {
            auto data = batch.data;
            auto target = batch.target.squeeze();

            data = data.to(torch::kF32);
            target = target.to(torch::kInt64);

            optimizer.zero_grad();

            auto output = net->forward(data);
            auto loss = torch::nll_loss(output, target);

            loss.backward();
            optimizer.step();

            std::cout 
                << "Train Epoch: "
                << epoch
                << " Loss: "
                << loss.item<float>()
                << std::endl;
            
        }
    }

    torch::save(net, "/home/joao/Documentos/dev/C++/test-pytorch/data/model.pt");    

    return 0;
}