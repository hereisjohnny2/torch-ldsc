#include <iostream>
#include <string>
#include <iomanip>
#include <memory>

#include <torch/optim.h>

#include "RockImageRGBDataset/RockImageRGBDataset.hpp"
#include "RockImageRGBNet/RockImageRGBNet.hpp"
#include "RockImageRGBTraining/RockImageRGBTraining.hpp"
#include "RockImageRGBTesting/RockImageRGBTesting.hpp"

int main(int argc, const char** argv) {
    double lr = 0.04;
    int batch_size = 5;

    if (argc == 3) {
        lr = std::atof(argv[1]);
        batch_size = std::atoi(argv[2]);
    }

    std::string filename = "/home/joao/Documentos/dev/C++/test-pytorch/data/training.dat";

    auto dataset = RockImageRGBDataset(filename).map(torch::data::transforms::Stack<>());
    auto dataLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(dataset),
        /*batch_size = */batch_size
    );
    auto datasetSize = dataset.size().value();

    auto net = std::make_shared<RockImageRGBNet>();
    torch::optim::SGD optimizer(net->parameters(), /*lr = */lr);

    RockImageRGBTraining train(net, optimizer);
    RockImageRGBTesting test(net, optimizer);

    for (int epoch = 0; epoch <= 1000; epoch++)
    {
        train.execute(epoch, datasetSize, *dataLoader);
        test.execute(epoch, datasetSize, *dataLoader);
    }

    torch::save(net, "/home/joao/Documentos/dev/C++/test-pytorch/data/model.pt");    

    return 0;
}