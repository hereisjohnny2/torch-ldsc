#include <iostream>
#include <string>
#include <iomanip>
#include <memory>

#include "RockImageRGBDataset/RockImageRGBDataset.hpp"
#include "RockImageRGBNet/RockImageRGBNet.hpp"
#include "utils/utils.hpp"

int main(int argc, const char** argv) {
    double lr = 0.04;
    int batch_size = 5;

    if (argc == 3) {
        lr = std::atof(argv[1]);
        batch_size = std::atoi(argv[2]);
    }


    std::string filename = "/home/joao/Documentos/dev/C++/test-pytorch/data/training.dat";
    std::string trainingData = utils::readDataFromFile(filename);

    auto dataset = RockImageRGBDataset(trainingData).map(torch::data::transforms::Stack<>());
    auto dataLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(dataset),
        /*batch_size = */batch_size
    );

    auto net = std::make_shared<RockImageRGBNet>();
    torch::optim::SGD optimizer(net->parameters(), /*lr = */lr);

    auto dataset_size = dataset.size().value();
    for (int epoch = 0; epoch <= 1000; epoch++)
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

    torch::save(net, "/home/joao/Documentos/dev/C++/test-pytorch/data/model.dat");    

    return 0;
}